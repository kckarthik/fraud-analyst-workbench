-- Phase 2 validation queries. Run with:
--   docker exec -i fraud-intel-db psql -U postgres -d fraud_intel < db\validate_phase2.sql

\echo '=== 1. Coverage: every alert should have all 4 enrichment keys ==='
SELECT COUNT(*) AS total,
       COUNT(*) FILTER (WHERE enrichment ? 'structured_facts')  AS sf,
       COUNT(*) FILTER (WHERE enrichment ? 'narrative_summary') AS narrative,
       COUNT(*) FILTER (WHERE enrichment ? 'model_score')       AS score,
       COUNT(*) FILTER (WHERE enrichment ? 'reason_codes')      AS reasons
FROM alerts;

\echo '=== 2. Does the model score separate fraud from non-fraud? (fraud avg should be much higher) ==='
SELECT d.decision,
       COUNT(*) AS n,
       ROUND(AVG((a.enrichment->>'model_score')::numeric), 5) AS avg_score,
       ROUND(MAX((a.enrichment->>'model_score')::numeric), 5) AS max_score
FROM alerts a
JOIN dispositions d ON d.alert_id = a.alert_id
GROUP BY d.decision;

\echo '=== 3. Fraud concentration in the top 1,000 ranked alerts (random would be ~0.5) ==='
SELECT COUNT(*) FILTER (WHERE d.decision = 'fraud') AS fraud_in_top_1000
FROM (
    SELECT alert_id
    FROM alerts
    WHERE enrichment ? 'model_score'
    ORDER BY (enrichment->>'model_score')::numeric DESC
    LIMIT 1000
) t
JOIN dispositions d ON d.alert_id = t.alert_id;

\echo '=== 4. Highest-scored actual fraud alert, with its SHAP reason codes ==='
SELECT jsonb_pretty(enrichment) AS enrichment
FROM alerts a
JOIN dispositions d ON d.alert_id = a.alert_id
WHERE d.decision = 'fraud'
ORDER BY (a.enrichment->>'model_score')::numeric DESC
LIMIT 1;
