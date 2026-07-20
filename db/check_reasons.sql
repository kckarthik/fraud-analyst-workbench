-- Expand an alert's SHAP reason_codes into a readable table.
--
-- Default: explains the single highest-scored alert.
-- To target a SPECIFIC alert, pass its id:
--   docker exec -i fraud-intel-db psql -U postgres -d fraud_intel -v aid=12345 < db\check_reasons.sql
-- (with no -v, it falls back to the top-scored alert)

\if :{?aid}
\else
  \set aid 0
\endif

SELECT
    a.alert_id,
    d.decision,
    ROUND((a.enrichment->>'model_score')::numeric, 4)      AS model_score,
    rc->>'feature'                                         AS feature,
    rc->>'value'                                           AS value,
    rc->>'direction'                                       AS direction,
    ROUND((rc->>'shap_value')::numeric, 3)                 AS shap_value,
    rc->>'description'                                      AS why
FROM alerts a
JOIN dispositions d ON d.alert_id = a.alert_id
CROSS JOIN LATERAL jsonb_array_elements(a.enrichment->'reason_codes') AS rc
WHERE a.alert_id = CASE
        WHEN :aid = 0
        THEN (SELECT alert_id FROM alerts WHERE enrichment ? 'model_score'
              ORDER BY (enrichment->>'model_score')::numeric DESC LIMIT 1)
        ELSE :aid
    END
ORDER BY abs((rc->>'shap_value')::numeric) DESC;
