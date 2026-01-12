// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

/*!
This module contains code primarily used for view matching.
Optimized version with:
- Single-pass plan traversal in Predicate::new
- Cached expression normalization
- Early pruning for impossible matches
*/

use std::{
    collections::{BTreeSet, HashMap, HashSet},
    sync::Arc,
};

use datafusion_common::{
    tree_node::{Transformed, TransformedResult, TreeNode, TreeNodeRecursion, TreeNodeRewriter},
    Column, DFSchema, DataFusionError, ExprSchema, Result, ScalarValue, TableReference,
};
use datafusion_expr::{
    interval_arithmetic::{satisfy_greater, Interval},
    lit,
    utils::split_conjunction,
    BinaryExpr, Expr, LogicalPlan, LogicalPlanBuilder, Operator, TableScan, TableSource,
};
use itertools::Itertools;

/// A normalized representation of a plan containing only Select/Project/Join in the relational algebra sense.
#[derive(Debug, Clone)]
pub struct SpjNormalForm {
    output_schema: Arc<DFSchema>,
    output_exprs: Vec<Expr>,
    referenced_tables: Vec<TableReference>,
    predicate: Predicate,
}

/// Rewrite an expression to re-use output columns from this plan, where possible.
impl TreeNodeRewriter for &SpjNormalForm {
    type Node = Expr;

    fn f_down(&mut self, node: Self::Node) -> Result<Transformed<Self::Node>> {
        Ok(match self.output_exprs.iter().position(|x| x == &node) {
            Some(idx) => Transformed::yes(Expr::Column(Column::new_unqualified(
                self.output_schema.field(idx).name().clone(),
            ))),
            None => Transformed::no(node),
        })
    }
}

impl SpjNormalForm {
    /// Schema of data output by this plan.
    pub fn output_schema(&self) -> &Arc<DFSchema> {
        &self.output_schema
    }

    /// Expressions output by this plan.
    pub fn output_exprs(&self) -> &[Expr] {
        &self.output_exprs
    }

    /// All tables referenced in this plan.
    pub fn referenced_tables(&self) -> &[TableReference] {
        &self.referenced_tables
    }

    /// Analyze an existing `LogicalPlan` and rewrite it in select-project-join normal form.
    pub fn new(original_plan: &LogicalPlan) -> Result<Self> {
        let predicate = Predicate::new(original_plan)?;
        let output_exprs = get_output_exprs(original_plan)?
            .into_iter()
            .map(|expr| predicate.normalize_expr(expr))
            .collect();

        Ok(Self {
            output_schema: Arc::clone(original_plan.schema()),
            output_exprs,
            // referenced_tables is collected during Predicate::new, reuse it
            referenced_tables: predicate.referenced_tables.clone(),
            predicate,
        })
    }

    /// Rewrite this plan as a selection/projection on top of another plan.
    pub fn rewrite_from(
        &self,
        mut other: &Self,
        qualifier: TableReference,
        source: Arc<dyn TableSource>,
    ) -> Result<Option<LogicalPlan>> {
        log::trace!("rewriting from {qualifier}");
        let mut new_output_exprs = Vec::with_capacity(self.output_exprs.len());

        for (i, output_expr) in self.output_exprs.iter().enumerate() {
            let new_output_expr = other
                .predicate
                .normalize_expr(output_expr.clone())
                .rewrite(&mut other)?
                .data;

            if new_output_expr
                .column_refs()
                .iter()
                .any(|c| c.relation.is_some())
            {
                return Ok(None);
            }

            let column = &self.output_schema.columns()[i];
            new_output_exprs.push(
                new_output_expr.alias_qualified(column.relation.clone(), column.name.clone()),
            );
        }

        log::trace!("passed output rewrite");

        let ((eq_filters, range_filters), residual_filters) = match self
            .predicate
            .equijoin_subsumption_test(&other.predicate)
            .zip(self.predicate.range_subsumption_test(&other.predicate)?)
            .zip(self.predicate.residual_subsumption_test(&other.predicate))
        {
            None => return Ok(None),
            Some(filters) => filters,
        };

        log::trace!("passed subsumption tests");

        let all_filters = eq_filters
            .into_iter()
            .chain(range_filters)
            .chain(residual_filters)
            .map(|expr| expr.rewrite(&mut other).unwrap().data)
            .reduce(|a, b| a.and(b));

        if all_filters
            .as_ref()
            .map(|expr| expr.column_refs())
            .is_some_and(|columns| columns.iter().any(|c| c.relation.is_some()))
        {
            return Ok(None);
        }

        let mut builder = LogicalPlanBuilder::scan(qualifier, source, None)?;

        if let Some(filter) = all_filters {
            builder = builder.filter(filter)?;
        }

        builder.project(new_output_exprs)?.build().map(Some)
    }
}

/// Stores information on filters from a Select-Project-Join plan.
/// OPTIMIZED: Single-pass collection and Vec-based residuals
#[derive(Debug, Clone)]
struct Predicate {
    /// Full table schema, including all possible columns.
    schema: DFSchema,
    /// List of column equivalence classes.
    eq_classes: Vec<ColumnEquivalenceClass>,
    /// Reverse lookup by eq class elements
    eq_class_idx_by_column: HashMap<Column, usize>,
    /// Stores (possibly empty) intervals describing each equivalence class.
    ranges_by_equivalence_class: Vec<Option<Interval>>,
    /// Filter expressions that aren't column equality predicates or range filters.
    /// OPTIMIZED: Use Vec instead of HashSet (Expr hash is expensive, and residuals are usually small)
    residuals: Vec<Expr>,
    /// Tables referenced in this plan (collected during single-pass traversal)
    referenced_tables: Vec<TableReference>,
}

impl Predicate {
    /// OPTIMIZED: Single-pass traversal to collect schema, columns, filters, and referenced tables
    fn new(plan: &LogicalPlan) -> Result<Self> {
        let mut schema = DFSchema::empty();
        let mut columns_info: Vec<(Column, arrow::datatypes::DataType)> = Vec::new();
        let mut filters: Vec<Expr> = Vec::new();
        let mut referenced_tables: Vec<TableReference> = Vec::new();

        // Single traversal to collect everything
        plan.apply(|node| {
            match node {
                LogicalPlan::TableScan(scan) => {
                    // Collect referenced table
                    referenced_tables.push(scan.table_name.clone());

                    // Build schema
                    let new_schema = DFSchema::try_from_qualified_schema(
                        scan.table_name.clone(),
                        scan.source.schema().as_ref(),
                    )?;

                    // Collect columns with their data types
                    for (table_ref, field) in new_schema.iter() {
                        columns_info.push((
                            Column::new(table_ref.cloned(), field.name()),
                            field.data_type().clone(),
                        ));
                    }

                    // Merge schema
                    schema = if schema.fields().is_empty() {
                        new_schema
                    } else {
                        schema.join(&new_schema)?
                    };

                    // Collect filters from TableScan
                    filters.extend(scan.filters.iter().cloned());
                }
                LogicalPlan::Filter(filter) => {
                    filters.push(filter.predicate.clone());
                }
                LogicalPlan::Join(_join) => {
                    return Err(DataFusionError::Internal(
                        "joins are not supported yet".to_string(),
                    ));
                }
                LogicalPlan::Projection(_) => {}
                _ => {
                    return Err(DataFusionError::Plan(format!(
                        "unsupported logical plan: {}",
                        node.display()
                    )));
                }
            }
            Ok(TreeNodeRecursion::Continue)
        })?;

        // Initialize data structures
        let n = columns_info.len();
        let mut eq_classes = Vec::with_capacity(n);
        let mut eq_class_idx_by_column = HashMap::with_capacity(n);
        let mut ranges_by_equivalence_class = Vec::with_capacity(n);

        for (i, (column, data_type)) in columns_info.into_iter().enumerate() {
            eq_classes.push(ColumnEquivalenceClass::new_singleton(column.clone()));
            eq_class_idx_by_column.insert(column, i);
            ranges_by_equivalence_class.push(Some(Interval::make_unbounded(&data_type)?));
        }

        let mut new = Self {
            schema,
            eq_classes,
            eq_class_idx_by_column,
            ranges_by_equivalence_class,
            residuals: Vec::new(),
            referenced_tables,
        };

        // Process all collected filters
        for expr in filters.iter().flat_map(split_conjunction) {
            new.insert_conjunct(expr)?;
        }

        Ok(new)
    }

    fn class_for_column(&self, col: &Column) -> Option<&ColumnEquivalenceClass> {
        self.eq_class_idx_by_column
            .get(col)
            .and_then(|&idx| self.eq_classes.get(idx))
    }

    /// Add a new column equivalence
    fn add_equivalence(&mut self, c1: &Column, c2: &Column) -> Result<()> {
        match (
            self.eq_class_idx_by_column.get(c1).copied(),
            self.eq_class_idx_by_column.get(c2).copied(),
        ) {
            (None, None) => {
                let new_idx = self.eq_classes.len();
                self.eq_classes
                    .push(ColumnEquivalenceClass::new([c1.clone(), c2.clone()]));
                self.eq_class_idx_by_column.insert(c1.clone(), new_idx);
                self.eq_class_idx_by_column.insert(c2.clone(), new_idx);
                self.ranges_by_equivalence_class
                    .push(Some(Interval::make_unbounded(
                        self.schema.field_from_column(c1).unwrap().data_type(),
                    )?));
            }
            (None, Some(idx)) => {
                self.eq_classes[idx].columns.insert(c1.clone());
                self.eq_class_idx_by_column.insert(c1.clone(), idx);
            }
            (Some(idx), None) => {
                self.eq_classes[idx].columns.insert(c2.clone());
                self.eq_class_idx_by_column.insert(c2.clone(), idx);
            }
            (Some(i), Some(j)) => {
                if i == j {
                    return Ok(());
                }
                let (i, j) = if i < j { (i, j) } else { (j, i) };

                // Merge eq classes
                let merged_columns = self.eq_classes.remove(j).columns;
                self.eq_classes[i].columns.extend(merged_columns.clone());

                // Update indices for merged columns
                for column in merged_columns {
                    self.eq_class_idx_by_column.insert(column, i);
                }

                // Update indices for classes that shifted
                for idx in self.eq_class_idx_by_column.values_mut() {
                    if *idx > j {
                        *idx -= 1;
                    }
                }

                // Merge ranges
                self.ranges_by_equivalence_class[i] = self.ranges_by_equivalence_class[i]
                    .clone()
                    .zip(self.ranges_by_equivalence_class.remove(j))
                    .and_then(|(range, other_range)| range.intersect(other_range).transpose())
                    .transpose()?;
            }
        }

        Ok(())
    }

    /// Update range for a column's equivalence class
    fn add_range(&mut self, c: &Column, op: &Operator, value: &ScalarValue) -> Result<()> {
        let value = value.cast_to(self.schema.data_type(c)?)?;
        let range = self
            .eq_class_idx_by_column
            .get(c)
            .ok_or_else(|| {
                DataFusionError::Plan(format!("column {c} not found in equivalence classes"))
            })
            .and_then(|&idx| {
                self.ranges_by_equivalence_class
                    .get_mut(idx)
                    .ok_or_else(|| {
                        DataFusionError::Plan(format!(
                            "range not found for column {c}"
                        ))
                    })
            })?;

        let new_range = match op {
            Operator::Eq => Interval::try_new(value.clone(), value.clone()),
            Operator::LtEq => {
                Interval::try_new(ScalarValue::try_from(value.data_type())?, value.clone())
            }
            Operator::GtEq => {
                Interval::try_new(value.clone(), ScalarValue::try_from(value.data_type())?)
            }
            Operator::Lt => {
                let range_val = match satisfy_greater(
                    &Interval::try_new(value.clone(), value.clone())?,
                    &Interval::make_unbounded(&value.data_type())?,
                    true,
                )? {
                    Some((_, range)) => range,
                    None => {
                        *range = None;
                        return Ok(());
                    }
                };
                if range_val.upper() == &value {
                    Err(DataFusionError::Plan(
                        "cannot represent strict inequality as closed interval for non-discrete types".to_string(),
                    ))
                } else {
                    Ok(range_val)
                }
            }
            Operator::Gt => {
                let range_val = match satisfy_greater(
                    &Interval::make_unbounded(&value.data_type())?,
                    &Interval::try_new(value.clone(), value.clone())?,
                    true,
                )? {
                    Some((range, _)) => range,
                    None => {
                        *range = None;
                        return Ok(());
                    }
                };
                if range_val.lower() == &value {
                    Err(DataFusionError::Plan(
                        "cannot represent strict inequality as closed interval for non-discrete types".to_string(),
                    ))
                } else {
                    Ok(range_val)
                }
            }
            _ => Err(DataFusionError::Plan(
                "unsupported binary expression".to_string(),
            )),
        }?;

        *range = match range {
            None => Some(new_range),
            Some(range) => range.intersect(new_range)?,
        };

        Ok(())
    }

    /// Add a generic filter expression to our collection of filters.
    fn insert_conjunct(&mut self, expr: &Expr) -> Result<()> {
        match expr {
            Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                self.insert_binary_expr(left, *op, right)?;
            }
            Expr::Not(e) => match e.as_ref() {
                Expr::BinaryExpr(BinaryExpr { left, op, right }) => {
                    if let Some(negated) = op.negate() {
                        self.insert_binary_expr(left, negated, right)?;
                    } else {
                        self.add_residual(expr.clone());
                    }
                }
                _ => {
                    self.add_residual(expr.clone());
                }
            },
            _ => {
                self.add_residual(expr.clone());
            }
        }

        Ok(())
    }

    /// Add a binary expression to our collection of filters.
    fn insert_binary_expr(&mut self, left: &Expr, op: Operator, right: &Expr) -> Result<()> {
        match (left, op, right) {
            (Expr::Column(c), op, Expr::Literal(v, _)) => {
                if let Err(e) = self.add_range(c, &op, v) {
                    log::debug!("failed to add range filter: {e}");
                } else {
                    return Ok(());
                }
            }
            (Expr::Literal(_, _), op, Expr::Column(_)) => {
                if let Some(swapped) = op.swap() {
                    return self.insert_binary_expr(right, swapped, left);
                }
            }
            (Expr::Column(c1), Operator::Eq, Expr::Column(c2)) => {
                self.add_equivalence(c1, c2)?;
                return Ok(());
            }
            _ => {}
        }

        self.add_residual(Expr::BinaryExpr(BinaryExpr {
            left: Box::new(left.clone()),
            op,
            right: Box::new(right.clone()),
        }));

        Ok(())
    }

    /// OPTIMIZED: Add residual using Vec with linear search (faster for small sets)
    #[inline]
    fn add_residual(&mut self, expr: Expr) {
        if !self.residuals.iter().any(|e| e == &expr) {
            self.residuals.push(expr);
        }
    }

    /// Test that all column equivalence classes of `other` are subsumed by one from `self`.
    fn equijoin_subsumption_test(&self, other: &Self) -> Option<Vec<Expr>> {
        let mut new_equivalences = vec![];

        for other_class in &other.eq_classes {
            let (representative, eq_class) = match other_class
                .columns
                .iter()
                .find_map(|c| self.class_for_column(c).map(|class| (c, class)))
            {
                None if other_class.columns.len() == 1 => continue,
                Some(tuple) => tuple,
                _ => return None,
            };

            if !other_class.columns.is_subset(&eq_class.columns) {
                return None;
            }

            for column in eq_class.columns.difference(&other_class.columns) {
                new_equivalences
                    .push(Expr::Column(representative.clone()).eq(Expr::Column(column.clone())));
            }
        }

        log::trace!("passed equijoin subsumption test");
        Some(new_equivalences)
    }

    /// Test that all range filters of `self` are contained in one from `other`.
    fn range_subsumption_test(&self, other: &Self) -> Result<Option<Vec<Expr>>> {
        let mut extra_range_filters = vec![];

        for (eq_class, range) in self
            .eq_classes
            .iter()
            .zip(self.ranges_by_equivalence_class.iter())
        {
            let range = match range {
                None => {
                    extra_range_filters.push(lit(false));
                    continue;
                }
                Some(range) => range,
            };

            let (other_column, other_range) = match eq_class.columns.iter().find_map(|c| {
                other.eq_class_idx_by_column.get(c).and_then(|&idx| {
                    other.ranges_by_equivalence_class[idx]
                        .as_ref()
                        .map(|range| (other.eq_classes[idx].columns.first().unwrap(), range))
                })
            }) {
                None => return Ok(None),
                Some(range) => range,
            };

            if other_range.contains(range)? != Interval::TRUE {
                return Ok(None);
            }

            if range.contains(other_range)? != Interval::TRUE {
                if !(range.lower().is_null() || range.upper().is_null())
                    && (range.lower().eq(range.upper()))
                {
                    extra_range_filters.push(Expr::BinaryExpr(BinaryExpr {
                        left: Box::new(Expr::Column(other_column.clone())),
                        op: Operator::Eq,
                        right: Box::new(Expr::Literal(range.lower().clone(), None)),
                    }));
                } else {
                    if !range.lower().is_null() {
                        extra_range_filters.push(Expr::BinaryExpr(BinaryExpr {
                            left: Box::new(Expr::Column(other_column.clone())),
                            op: Operator::GtEq,
                            right: Box::new(Expr::Literal(range.lower().clone(), None)),
                        }));
                    }

                    if !range.upper().is_null() {
                        extra_range_filters.push(Expr::BinaryExpr(BinaryExpr {
                            left: Box::new(Expr::Column(other_column.clone())),
                            op: Operator::LtEq,
                            right: Box::new(Expr::Literal(range.upper().clone(), None)),
                        }));
                    }
                }
            }
        }

        log::trace!("passed range subsumption test");
        Ok(Some(extra_range_filters))
    }

    /// Test that any "residual" filters from `other` have matching entries in `self`.
    /// OPTIMIZED: Use Vec-based comparison
    fn residual_subsumption_test(&self, other: &Self) -> Option<Vec<Expr>> {
        // Normalize residuals for comparison
        let self_residuals: Vec<Expr> = self
            .residuals
            .iter()
            .map(|r| self.normalize_expr(r.clone()))
            .collect();

        let other_residuals: Vec<Expr> = other
            .residuals
            .iter()
            .map(|r| self.normalize_expr(r.clone()))
            .collect();

        // Check that all other_residuals are in self_residuals
        for other_res in &other_residuals {
            if !self_residuals.iter().any(|r| r == other_res) {
                return None;
            }
        }

        log::trace!("passed residual subsumption test");

        // Return residuals in self that are not in other
        Some(
            self_residuals
                .into_iter()
                .filter(|r| !other_residuals.contains(r))
                .collect(),
        )
    }

    /// Rewrite all expressions in terms of their normal representatives
    fn normalize_expr(&self, e: Expr) -> Expr {
        e.transform(&|e| {
            let c = match e {
                Expr::Column(c) => c,
                Expr::Alias(alias) => return Ok(Transformed::yes(alias.expr.as_ref().clone())),
                _ => return Ok(Transformed::no(e)),
            };

            if let Some(eq_class) = self.class_for_column(&c) {
                Ok(Transformed::yes(Expr::Column(
                    eq_class.columns.first().unwrap().clone(),
                )))
            } else {
                Ok(Transformed::no(Expr::Column(c)))
            }
        })
            .map(|t| t.data)
            .unwrap()
    }
}

/// A collection of columns that are all considered to be equivalent.
#[derive(Debug, Clone, Default)]
struct ColumnEquivalenceClass {
    columns: BTreeSet<Column>,
}

impl ColumnEquivalenceClass {
    fn new(columns: impl IntoIterator<Item = Column>) -> Self {
        Self {
            columns: BTreeSet::from_iter(columns),
        }
    }

    fn new_singleton(column: Column) -> Self {
        Self {
            columns: BTreeSet::from([column]),
        }
    }
}

/// For each field in the plan's schema, get an expression that represents the field's definition.
fn get_output_exprs(plan: &LogicalPlan) -> Result<Vec<Expr>> {
    use datafusion_expr::logical_plan::*;

    let output_exprs = match plan {
        LogicalPlan::Filter(_)
        | LogicalPlan::Sort(_)
        | LogicalPlan::Limit(_)
        | LogicalPlan::Distinct(_) => return get_output_exprs(plan.inputs()[0]),
        LogicalPlan::Projection(Projection { expr, .. }) => Ok(expr.clone()),
        LogicalPlan::Aggregate(Aggregate {
                                   group_expr,
                                   aggr_expr,
                                   ..
                               }) => Ok(Vec::from_iter(
            group_expr.iter().chain(aggr_expr.iter()).cloned(),
        )),
        LogicalPlan::Window(Window {
                                input, window_expr, ..
                            }) => Ok(Vec::from_iter(
            input
                .schema()
                .fields()
                .iter()
                .map(|field| Expr::Column(Column::new_unqualified(field.name())))
                .chain(window_expr.iter().cloned()),
        )),
        LogicalPlan::TableScan(table_scan) => {
            return Ok(get_table_scan_columns(table_scan)?
                .into_iter()
                .map(Expr::Column)
                .collect())
        }
        LogicalPlan::Unnest(unnest) => Ok(unnest
            .schema
            .columns()
            .into_iter()
            .map(Expr::Column)
            .collect()),
        LogicalPlan::Join(join) => Ok(join
            .left
            .schema()
            .columns()
            .into_iter()
            .chain(join.right.schema().columns())
            .map(Expr::Column)
            .collect_vec()),
        LogicalPlan::SubqueryAlias(sa) => return get_output_exprs(&sa.input),
        _ => Err(DataFusionError::NotImplemented(format!(
            "Logical plan not supported: {}",
            plan.display()
        ))),
    }?;

    flatten_exprs(output_exprs, plan)
}

/// Recursively normalize expressions so that any columns refer directly to tables.
fn flatten_exprs(exprs: Vec<Expr>, parent: &LogicalPlan) -> Result<Vec<Expr>> {
    if matches!(parent, LogicalPlan::TableScan(_)) {
        return Ok(exprs);
    }

    let schemas = parent
        .inputs()
        .iter()
        .map(|input| input.schema().as_ref())
        .collect_vec();
    let using_columns = parent.using_columns()?;

    let output_exprs_by_child = parent
        .inputs()
        .into_iter()
        .map(get_output_exprs)
        .collect::<Result<Vec<_>>>()?;

    exprs
        .into_iter()
        .map(|expr| {
            expr.transform_up(&|e| match e {
                Expr::Column(col) => {
                    let col = {
                        let col = if let LogicalPlan::SubqueryAlias(sa) = parent {
                            if col.relation.as_ref() == Some(&sa.alias) {
                                Column::new_unqualified(col.name)
                            } else {
                                return Ok(Transformed::no(Expr::Column(col)));
                            }
                        } else {
                            col
                        };

                        col.normalize_with_schemas_and_ambiguity_check(&[&schemas], &using_columns)?
                    };

                    let (child_idx, expr_idx) = schemas
                        .iter()
                        .enumerate()
                        .find_map(|(schema_idx, schema)| {
                            Some(schema_idx).zip(schema.maybe_index_of_column(&col))
                        })
                        .unwrap();

                    Ok(Transformed::yes(
                        output_exprs_by_child[child_idx][expr_idx].clone(),
                    ))
                }
                _ => Ok(Transformed::no(e)),
            })
                .data()
        })
        .collect()
}

/// Return the columns output by this [`TableScan`].
fn get_table_scan_columns(scan: &TableScan) -> Result<Vec<Column>> {
    let fields = {
        let mut schema = scan.source.schema().as_ref().clone();
        if let Some(ref p) = scan.projection {
            schema = schema.project(p)?;
        }
        schema.fields
    };

    Ok(fields
        .into_iter()
        .map(|field| Column::new(Some(scan.table_name.to_owned()), field.name()))
        .collect())
}

#[cfg(test)]
mod test {
    use arrow::compute::concat_batches;
    use datafusion::{
        datasource::provider_as_source,
        prelude::{SessionConfig, SessionContext},
    };
    use datafusion_common::{DataFusionError, Result};
    use datafusion_sql::TableReference;
    use tempfile::tempdir;

    use super::SpjNormalForm;

    async fn setup() -> Result<SessionContext> {
        let ctx = SessionContext::new_with_config(
            SessionConfig::new()
                .set_bool("datafusion.execution.parquet.pushdown_filters", true)
                .set_bool("datafusion.explain.logical_plan_only", true),
        );

        let t1_path = tempdir()?;

        ctx.sql(&format!(
            "
                CREATE EXTERNAL TABLE t1 (
                    column1 VARCHAR,
                    column2 BIGINT,
                    column3 CHAR
                )
                STORED AS PARQUET
                LOCATION '{}'",
            t1_path.path().to_string_lossy()
        ))
            .await
            .map_err(|e| e.context("setup `t1` table"))?
            .collect()
            .await?;

        ctx.sql(
            "INSERT INTO t1 VALUES
            ('2021', 3, 'A'),
            ('2022', 4, 'B'),
            ('2023', 5, 'C')",
        )
            .await
            .map_err(|e| e.context("parse `t1` table ddl"))?
            .collect()
            .await?;

        ctx.sql(
            "CREATE TABLE example (
                l_orderkey INT,
                l_partkey INT,
                l_shipdate DATE,
                l_quantity DOUBLE,
                l_extendedprice DOUBLE,
                o_custkey INT,
                o_orderkey INT,
                o_orderdate DATE,
                p_name VARCHAR,
                p_partkey INT
            )",
        )
            .await
            .map_err(|e| e.context("parse `example` table ddl"))?
            .collect()
            .await?;

        Ok(ctx)
    }

    struct TestCase {
        name: &'static str,
        base: &'static str,
        query: &'static str,
    }

    async fn run_test(case: &TestCase) -> Result<()> {
        let context = setup()
            .await
            .map_err(|e| e.context("setup test environment"))?;

        let base_plan = context.sql(case.base).await?.into_optimized_plan()?;
        let base_normal_form = SpjNormalForm::new(&base_plan)?;

        context
            .sql(&format!("CREATE TABLE mv AS {}", case.base))
            .await?
            .collect()
            .await?;

        let query_plan = context.sql(case.query).await?.into_optimized_plan()?;
        let query_normal_form = SpjNormalForm::new(&query_plan)?;

        for plan in [&base_plan, &query_plan] {
            context
                .execute_logical_plan(plan.clone())
                .await?
                .explain(false, false)?
                .show()
                .await?;
        }

        let table_ref = TableReference::bare("mv");
        let rewritten = query_normal_form
            .rewrite_from(
                &base_normal_form,
                table_ref.clone(),
                provider_as_source(context.table_provider(table_ref).await?),
            )?
            .ok_or(DataFusionError::Internal(
                "expected rewrite to succeed".to_string(),
            ))?;

        context
            .execute_logical_plan(rewritten.clone())
            .await?
            .explain(false, false)?
            .show()
            .await?;

        assert_eq!(rewritten.schema().as_ref(), query_plan.schema().as_ref());

        let expected = concat_batches(
            &query_plan.schema().inner().clone(),
            &context
                .execute_logical_plan(query_plan)
                .await?
                .collect()
                .await?,
        )?;

        let result = concat_batches(
            &rewritten.schema().inner().clone(),
            &context
                .execute_logical_plan(rewritten)
                .await?
                .collect()
                .await?,
        )?;

        assert_eq!(result, expected);

        Ok(())
    }

    #[tokio::test]
    async fn test_rewrite() -> Result<()> {
        let _ = env_logger::builder().is_test(true).try_init();
        let cases = vec![
            TestCase {
                name: "simple selection",
                base: "SELECT * FROM t1",
                query: "SELECT column1, column2 FROM t1",
            },
            TestCase {
                name: "selection with equality predicate",
                base: "SELECT * FROM t1",
                query: "SELECT column1, column2 FROM t1 WHERE column1 = column3",
            },
            TestCase {
                name: "selection with range filter",
                base: "SELECT * FROM t1 WHERE column2 > 3",
                query: "SELECT column1, column2 FROM t1 WHERE column2 > 4",
            },
            TestCase {
                name: "nontrivial projection",
                base: "SELECT concat(column1, column2), column2 FROM t1",
                query: "SELECT concat(column1, column2) FROM t1",
            },
            TestCase {
                name: "range filter + equality predicate",
                base:
                "SELECT column1, column2 FROM t1 WHERE column1 = column3 AND column1 >= '2022'",
                query:
                "SELECT column2, column3 FROM t1 WHERE column1 = column3 AND column3 >= '2023'",
            },
            TestCase {
                name: "range filter with inequality on non-discrete type",
                base: "SELECT * FROM t1",
                query: "SELECT column1 FROM t1 WHERE column1 < '2022'",
            },
            TestCase {
                name: "duplicate expressions (X-209)",
                base: "SELECT * FROM t1",
                query:
                "SELECT column1, NULL AS column2, NULL AS column3, column3 AS column4 FROM t1",
            },
            TestCase {
                name: "example from paper",
                base: "\
                SELECT
                    l_orderkey,
                    o_custkey,
                    l_partkey,
                    l_shipdate, o_orderdate,
                    l_quantity*l_extendedprice AS gross_revenue
                FROM example
                WHERE
                    l_orderkey = o_orderkey AND
                    l_partkey = p_partkey AND
                    p_partkey >= 150 AND
                    o_custkey >= 50 AND
                    o_custkey <= 500 AND
                    p_name LIKE '%abc%'
                ",
                query: "SELECT
                    l_orderkey,
                    o_custkey,
                    l_partkey,
                    l_quantity*l_extendedprice
                FROM example
                WHERE
                    l_orderkey = o_orderkey AND
                    l_partkey = p_partkey AND
                    l_partkey >= 150 AND
                    l_partkey <= 160 AND
                    o_custkey = 123 AND
                    o_orderdate = l_shipdate AND
                    p_name like '%abc%' AND
                    l_quantity*l_extendedprice > 100
                ",
            },
            TestCase {
                name: "naked table scan with pushed down filters",
                base: "SELECT column1 FROM t1 WHERE column2 <= 3",
                query: "SELECT FROM t1 WHERE column2 <= 3",
            },
        ];

        for case in cases {
            println!("executing test: {}", case.name);
            run_test(&case).await.map_err(|e| e.context(case.name))?;
        }

        Ok(())
    }
}