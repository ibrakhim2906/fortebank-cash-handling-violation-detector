# Development errors — measured system results

All 40 supplied recordings were previously reviewed. Reserve is prospective regression only; not independent. Setup metadata is team supplied.

Engines: {"mission-v1": 28, "missing_trace": 2}. Recipe hashes: ["1f95e816f2e04e23452d2c6403518855b26eea7dd5a82ea1278df8b2f7a99672"].

| Video | Human target types | Status / predicted types | Time hits | Engine | Auditor / failure |
|---|---|---|---|---|---|
|Х-17|hand_to_hand_transfer,unauthorized_access|done / none|0/1|mission-v1|:clear; personal_items:not_visible|
|Х-30|door_left_open|done / none|0/1|mission-v1|:clear|
|Х-19|clean|done / food_drink|0/0|mission-v1|:clear; food_drink:clear|
|Х-48,Х-5|clean|done / cash_left_out|0/0|mission-v1|:clear; cash_left_out:clear|
|Х-33|unauthorized_access|done / none|0/1|mission-v1|:clear|
|Х-3|clean|done / none|0/0|mission-v1|:clear; counting_without_client:not_visible|
|Х-35|clean|done / none|0/0|mission-v1|:clear|
|Х-45|clean|failed / none|0/0|missing||
|Х-10|clean|done / none|0/0|mission-v1|:clear|
|Х-38|cash_left_out|done / none|0/1|mission-v1|:clear; cash_left_out:not_visible|
|Х-46|food_drink|done / food_drink|1/1|mission-v1|:clear; food_drink:clear|
|Х-15|cash_left_out|done / personal_items|0/1|mission-v1|:clear; personal_items:clear|
|Х-8|clean|done / none|0/0|mission-v1|:clear|
|Х-40|personal_items|done / none|0/1|mission-v1|:clear|
|Х-14|clean|done / cash_left_out|0/0|mission-v1|:clear; cash_left_out:clear|
|X-1|no_receipt|done / none|0/1|mission-v1|:clear; food_drink:not_visible|
|Х-25|clean|done / none|0/0|mission-v1|:clear; safe_left_open:not_visible|
|Х-18|hand_to_hand_transfer,unauthorized_access|failed / none|0/1|missing||
|Х-12|clean|done / none|0/0|mission-v1|:clear|
|X-2|no_id_check|done / none|0/1|mission-v1|:clear|
|Х-13|cash_left_out,safe_left_open|done / none|0/1|mission-v1|:clear|
|Х-11|door_left_open|done / none|0/1|mission-v1|:clear|
|Х-29|clean|done / none|0/0|mission-v1|:clear|
|Х-22|keys_left_in_safe|done / none|0/1|mission-v1|:clear|
|Х-28|workstation_unlocked|done / none|0/1|mission-v1|:clear|
|Х-27|clean|done / none|0/0|mission-v1|:clear|
|Х-39|clean|done / none|0/0|mission-v1|:clear|
|Х-26|cash_left_out|done / none|0/1|mission-v1|:clear|
|Х-21|clean|done / none|0/0|mission-v1|:clear|
|Х-34|counting_without_client|done / none|0/1|mission-v1|:clear|

Clusters (counts can overlap for multi-type events): [["missed:cash_left_out", 3], ["missed:unauthorized_access", 2], ["missed:door_left_open", 2], ["false_alarm:cash_left_out", 2], ["processing_failure", 2], ["missed:hand_to_hand_transfer", 1], ["false_alarm:food_drink", 1], ["localization_miss", 1], ["missed:personal_items", 1], ["missed:no_receipt", 1], ["missed:no_id_check", 1], ["missed:safe_left_open", 1], ["missed:keys_left_in_safe", 1], ["missed:workstation_unlocked", 1], ["missed:counting_without_client", 1], ["type_mismatch", 0]]
