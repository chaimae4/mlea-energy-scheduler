# ML-EA: Energy-Aware VM Scheduling

Code for the paper "A hybrid machine learning framework for energy-aware
virtual machine scheduling in cloud computing".

## Reproduce the results

1. Extract GCT-v3 tasks via BigQuery (see `requete_bigquery_taches.sql`),
   save as `taches_gct.csv`
2. Install: `pip install -r requirements.txt`
3. Run: `python3 mlea_experiences.py`

## Datasets
- Google Cluster Trace v3: https://github.com/google/cluster-data
- Google PowerData 2019: `gs://powerdata_2019/`
- Alibaba Cluster Trace v2: https://github.com/alibaba/clusterdata

## License
MIT
