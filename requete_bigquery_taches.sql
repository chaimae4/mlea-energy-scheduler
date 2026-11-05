
WITH usage AS (
  SELECT
    collection_id,
    instance_index,
    MIN(start_time)              AS t_start,
    MAX(end_time)                AS t_end,
    AVG(average_usage.cpus)      AS cpu,
    AVG(average_usage.memory)    AS mem
  FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_usage`
  WHERE start_time < 7 * 24 * 3600 * 1000000
    AND average_usage.cpus   IS NOT NULL
    AND average_usage.memory IS NOT NULL
  GROUP BY collection_id, instance_index
),
events AS (
  SELECT
    collection_id,
    instance_index,
    ANY_VALUE(scheduling_class) AS sched_class,
    ANY_VALUE(priority)         AS priority
  FROM `google.com:google-cluster-data.clusterdata_2019_a.instance_events`
  WHERE time < 7 * 24 * 3600 * 1000000
  GROUP BY collection_id, instance_index
)
SELECT
  u.t_start,
  (u.t_end - u.t_start) / 1000000 AS duration_s,
  u.cpu,
  u.mem,
  e.sched_class,
  e.priority
FROM usage u
JOIN events e
  ON  u.collection_id  = e.collection_id
  AND u.instance_index = e.instance_index
WHERE u.t_end > u.t_start
ORDER BY RAND()
LIMIT 50000
