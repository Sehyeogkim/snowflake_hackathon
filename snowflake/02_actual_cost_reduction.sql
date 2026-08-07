-- Replace run_001 with the fixed benchmark ID used in QUERY_TAG.
SET RUN_ID = 'run_001';

WITH COSTS AS (
  SELECT
    QUERY_TAG,
    SUM(CREDITS) AS AI_CREDITS,
    COUNT(DISTINCT QUERY_ID) AS QUERY_COUNT
  FROM SNOWFLAKE.ACCOUNT_USAGE.CORTEX_AI_FUNCTIONS_USAGE_HISTORY
  WHERE QUERY_TAG LIKE 'mavis|baseline|' || $RUN_ID || '|%'
     OR QUERY_TAG LIKE 'mavis|optimized|' || $RUN_ID || '|%'
  GROUP BY QUERY_TAG
), PIVOTED AS (
  SELECT
    SUM(IFF(QUERY_TAG LIKE 'mavis|baseline|' || $RUN_ID || '|%', AI_CREDITS, 0))
      AS BASELINE_CREDITS,
    SUM(IFF(QUERY_TAG LIKE 'mavis|optimized|' || $RUN_ID || '|%', AI_CREDITS, 0))
      AS OPTIMIZED_CREDITS,
    SUM(IFF(QUERY_TAG LIKE 'mavis|baseline|' || $RUN_ID || '|%', QUERY_COUNT, 0))
      AS BASELINE_QUERIES,
    SUM(IFF(QUERY_TAG LIKE 'mavis|optimized|' || $RUN_ID || '|%', QUERY_COUNT, 0))
      AS OPTIMIZED_QUERIES
  FROM COSTS
)
SELECT
  BASELINE_CREDITS,
  OPTIMIZED_CREDITS,
  BASELINE_QUERIES,
  OPTIMIZED_QUERIES,
  100 * (BASELINE_CREDITS - OPTIMIZED_CREDITS)
    / NULLIF(BASELINE_CREDITS, 0) AS ACTUAL_COST_REDUCTION_PCT
FROM PIVOTED;

-- Usage history can lag by up to ~5 minutes. Long calls may span multiple hourly
-- rows, so SUM every row; do not filter to IS_COMPLETED=TRUE when summing credits.
