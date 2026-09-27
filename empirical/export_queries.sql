-- Saved aggregation queries. source_search is the prepared results-page table
-- for the four sampled weeks; the original query-to-export lineage is unavailable.

SELECT activity_class, season, COUNT(*) AS n_searches,
       AVG(clicks_count) AS avg_clicks,
       AVG(CAST(has_conversion AS DOUBLE)) AS conversion_rate,
       AVG(bounds_diag_km) AS avg_bounds_diag_km,
       AVG(answers_count) AS avg_answers_count,
       AVG(good_use_rate) AS avg_good_use_rate
FROM source_search
WHERE activity_class IS NOT NULL
GROUP BY activity_class, season
ORDER BY activity_class, season;

SELECT log_date, activity_class, COUNT(*) AS n_searches,
       AVG(clicks_count) AS avg_clicks,
       AVG(good_use_rate) AS avg_good_use_rate,
       AVG(answers_count) AS avg_answers
FROM source_search
WHERE season = 'spring'
  AND answers_count > 0 AND clicks_count > 0
  AND answer_lat BETWEEN 55.49 AND 56.02
  AND answer_lon BETWEEN 37.32 AND 37.97
  AND activity_class IN ('food','gas_stations','pharmacy','leisure','shopping')
GROUP BY log_date, activity_class
ORDER BY activity_class, log_date;

SELECT activity_class, COUNT(*) AS n_searches,
       AVG(answers_count) AS avg_answers,
       AVG(clicks_count) AS avg_clicks,
       AVG(good_use_rate) AS avg_good_use_rate,
       AVG(answer_lat) AS avg_answer_lat,
       AVG(answer_lon) AS avg_answer_lon,
       STDDEV(answer_lat) AS std_answer_lat,
       STDDEV(answer_lon) AS std_answer_lon
FROM source_search
WHERE log_date = '2025-04-14'
  AND answers_count > 0 AND clicks_count > 0
  AND has_conversion = TRUE AND good_use_rate > 0
  AND answer_lat BETWEEN 55.49 AND 56.02
  AND answer_lon BETWEEN 37.32 AND 37.97
  AND activity_class IN ('food','gas_stations','pharmacy','leisure','shopping')
GROUP BY activity_class
ORDER BY activity_class;
