# Amazon ML Challenge 2026 — Phase 1 Dataset EDA & Ground-Truth Report
**Execution Date**: 2026-09-26
**Scope**: Official closed challenge dataset under `student_resource/dataset/`
---

## 1. Dataset Dimensions & Memory Footprint

| Dataset | Rows | Columns | Shape | Memory (MB) |
| --- | --- | --- | --- | --- |
| train_source1 | 2206821 | 4 | (2206821, 4) | 671.78 |
| train_source2 | 5034616 | 4 | (5034616, 4) | 1580.64 |
| train_source3 | 5285603 | 4 | (5285603, 4) | 1646.7 |
| train_ground_truth | 2206821 | 2 | (2206821, 2) | 353.91 |
| test_source1 | 1732544 | 4 | (1732544, 4) | 539.63 |
| test_source2 | 4887273 | 4 | (4887273, 4) | 1580.52 |
| test_source3 | 5082316 | 4 | (5082316, 4) | 1617.69 |


## 2. Column Structure & Schema Validation

### `train_source1`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'entity_id': 'object', 'business_name': 'object', 'business_address': 'object', 'country': 'object'}`

### `train_source2`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'entity_id': 'object', 'business_name': 'object', 'business_address': 'object', 'country': 'object'}`

### `train_source3`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'entity_id': 'object', 'business_name': 'object', 'business_address': 'object', 'country': 'object'}`

### `train_ground_truth`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'source1_entity_id': 'object', 'matched_entity_ids': 'object'}`

### `test_source1`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'entity_id': 'object', 'business_name': 'object', 'business_address': 'object', 'country': 'object'}`

### `test_source2`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'entity_id': 'object', 'business_name': 'object', 'business_address': 'object', 'country': 'object'}`

### `test_source3`
- **Valid Schema**: `PASS`
- **Columns & Dtypes**: `{'entity_id': 'object', 'business_name': 'object', 'business_address': 'object', 'country': 'object'}`


## 3. Missing-Value Analysis

| Dataset | Column | Total Rows | Missing Count | Missing Pct (%) |
| --- | --- | --- | --- | --- |
| train_source1 | entity_id | 2206821 | 0 | 0.0 |
| train_source1 | business_name | 2206821 | 0 | 0.0 |
| train_source1 | business_address | 2206821 | 0 | 0.0 |
| train_source1 | country | 2206821 | 0 | 0.0 |
| train_source2 | entity_id | 5034616 | 0 | 0.0 |
| train_source2 | business_name | 5034616 | 2 | 0.0 |
| train_source2 | business_address | 5034616 | 168967 | 3.3561 |
| train_source2 | country | 5034616 | 0 | 0.0 |
| train_source3 | entity_id | 5285603 | 0 | 0.0 |
| train_source3 | business_name | 5285603 | 13 | 0.0002 |
| train_source3 | business_address | 5285603 | 175916 | 3.3282 |
| train_source3 | country | 5285603 | 0 | 0.0 |
| train_ground_truth | source1_entity_id | 2206821 | 0 | 0.0 |
| train_ground_truth | matched_entity_ids | 2206821 | 123247 | 5.5848 |
| test_source1 | entity_id | 1732544 | 0 | 0.0 |
| test_source1 | business_name | 1732544 | 0 | 0.0 |
| test_source1 | business_address | 1732544 | 0 | 0.0 |
| test_source1 | country | 1732544 | 0 | 0.0 |
| test_source2 | entity_id | 4887273 | 0 | 0.0 |
| test_source2 | business_name | 4887273 | 46 | 0.0009 |
| test_source2 | business_address | 4887273 | 129408 | 2.6479 |
| test_source2 | country | 4887273 | 0 | 0.0 |
| test_source3 | entity_id | 5082316 | 0 | 0.0 |
| test_source3 | business_name | 5082316 | 59 | 0.0012 |
| test_source3 | business_address | 5082316 | 136098 | 2.6779 |
| test_source3 | country | 5082316 | 0 | 0.0 |


## 4. Duplicate Analysis

| Dataset | Check | Total Rows | Duplicate Count | Duplicate Pct (%) |
| --- | --- | --- | --- | --- |
| train_source1 | entity_id | 2206821 | 0 | 0.0 |
| train_source1 | business_name | 2206821 | 667592 | 30.2513 |
| train_source1 | business_address | 2206821 | 76215 | 3.4536 |
| train_source1 | name + address | 2206821 | 0 | 0.0 |
| train_source2 | entity_id | 5034616 | 0 | 0.0 |
| train_source2 | business_name | 5034616 | 632606 | 12.5651 |
| train_source2 | business_address | 5034616 | 528388 | 10.4951 |
| train_source2 | name + address | 5034616 | 25891 | 0.5143 |
| train_source3 | entity_id | 5285603 | 0 | 0.0 |
| train_source3 | business_name | 5285603 | 633982 | 11.9945 |
| train_source3 | business_address | 5285603 | 476923 | 9.0231 |
| train_source3 | name + address | 5285603 | 18881 | 0.3572 |
| train_ground_truth | source1_entity_id | 2206821 | 0 | 0.0 |
| test_source1 | entity_id | 1732544 | 0 | 0.0 |
| test_source1 | business_name | 1732544 | 493677 | 28.4943 |
| test_source1 | business_address | 1732544 | 55061 | 3.178 |
| test_source1 | name + address | 1732544 | 0 | 0.0 |
| test_source2 | entity_id | 4887273 | 0 | 0.0 |
| test_source2 | business_name | 4887273 | 576187 | 11.7895 |
| test_source2 | business_address | 4887273 | 533082 | 10.9076 |
| test_source2 | name + address | 4887273 | 22642 | 0.4633 |
| test_source3 | entity_id | 5082316 | 0 | 0.0 |
| test_source3 | business_name | 5082316 | 560329 | 11.0251 |
| test_source3 | business_address | 5082316 | 489783 | 9.637 |
| test_source3 | name + address | 5082316 | 16305 | 0.3208 |


## 5. Country Distribution (Open-Set Analysis)

| Dataset | Country | Count | Percentage (%) |
| --- | --- | --- | --- |
| train_source1 | US | 1323633 | 59.9792 |
| train_source1 | India | 883188 | 40.0208 |
| train_source2 | US | 3016817 | 59.9215 |
| train_source2 | India | 2017799 | 40.0785 |
| train_source3 | US | 3170056 | 59.9753 |
| train_source3 | India | 2115547 | 40.0247 |
| test_source1 | India | 809986 | 46.7513 |
| test_source1 | US | 663106 | 38.2735 |
| test_source1 | France | 259452 | 14.9752 |
| test_source2 | India | 2312565 | 47.3181 |
| test_source2 | US | 1871330 | 38.2899 |
| test_source2 | France | 703378 | 14.392 |
| test_source3 | India | 2405000 | 47.3209 |
| test_source3 | US | 1945701 | 38.2837 |
| test_source3 | France | 731615 | 14.3953 |


> **Key Finding**: Country must be treated as an open-set categorical feature. Note the differences between training and test sets.

## 6. String Length & Whitespace Statistics

| Dataset | Column | Min Length | Max Length | Mean Length | Std Dev | Median (P50) | P5 | P25 | P75 | P95 | P99 | Empty/Whitespace Count | Empty/Whitespace Pct (%) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train_source1 | business_name | 3 | 105 | 24.03 | 7.74 | 24.0 | 12.0 | 18.0 | 30.0 | 37.0 | 42.0 | 0 | 0.0 |
| train_source1 | business_address | 11 | 256 | 52.07 | 25.33 | 41.0 | 27.0 | 33.0 | 70.0 | 103.0 | 124.0 | 0 | 0.0 |
| train_source2 | business_name | 0 | 104 | 25.1 | 8.89 | 25.0 | 12.0 | 19.0 | 31.0 | 40.0 | 48.0 | 2 | 0.0 |
| train_source2 | business_address | 0 | 249 | 46.23 | 24.84 | 37.0 | 22.0 | 30.0 | 61.0 | 96.0 | 118.0 | 168967 | 3.3561 |
| train_source3 | business_name | 0 | 123 | 25.2 | 9.49 | 25.0 | 11.0 | 18.0 | 31.0 | 42.0 | 50.0 | 13 | 0.0002 |
| train_source3 | business_address | 0 | 240 | 46.71 | 21.65 | 42.0 | 22.0 | 35.0 | 54.0 | 91.0 | 115.0 | 175916 | 3.3282 |
| test_source1 | business_name | 3 | 92 | 23.84 | 7.67 | 24.0 | 12.0 | 18.0 | 29.0 | 36.0 | 42.0 | 0 | 0.0 |
| test_source1 | business_address | 11 | 268 | 57.21 | 25.03 | 50.0 | 28.0 | 36.0 | 74.0 | 105.0 | 126.0 | 0 | 0.0 |
| test_source2 | business_name | 0 | 102 | 25.7 | 9.12 | 25.0 | 12.0 | 19.0 | 32.0 | 42.0 | 49.0 | 46 | 0.0009 |
| test_source2 | business_address | 0 | 269 | 50.41 | 25.35 | 43.0 | 23.0 | 32.0 | 67.0 | 99.0 | 120.0 | 129408 | 2.6479 |
| test_source3 | business_name | 0 | 103 | 25.66 | 9.57 | 25.0 | 11.0 | 19.0 | 32.0 | 42.0 | 50.0 | 59 | 0.0012 |
| test_source3 | business_address | 0 | 267 | 48.74 | 22.64 | 43.0 | 22.0 | 35.0 | 59.0 | 94.0 | 117.0 | 136098 | 2.6779 |


## 7. Ground-Truth Match Distribution

| Metric | Value |
| --- | --- |
| Total Source 1 Entities | 2206821.0 |
| Zero Matches (Singletons) | 123247.0 |
| Zero Matches Pct (%) | 5.5848 |
| Exactly 1 Match | 119157.0 |
| Exactly 1 Match Pct (%) | 5.3995 |
| Exactly 2 Matches | 375212.0 |
| Exactly 2 Matches Pct (%) | 17.0024 |
| 3+ Matches | 1589205.0 |
| 3+ Matches Pct (%) | 72.0133 |
| Mean Matches per S1 | 3.4613 |
| Median Matches per S1 | 3.0 |
| Max Matches per S1 | 11.0 |
| Total S2 Matches | 3693619.0 |
| Total S3 Matches | 3944746.0 |
| Avg S2 Matches per S1 | 1.6737 |
| Avg S3 Matches per S1 | 1.7875 |


## 8. Ground-Truth Integrity Verification

- **Every GT S1 ID exists in train_source1**: `True`
- **GT S1 IDs missing in train_source1 count**: `0`
- **Train S1 IDs missing in GT count**: `0`
- **GT row count matches train_source1 row count**: `True`
- **No Source 1 ID appears as matched ID**: `True`
- **Source 1 IDs as matched ID violation count**: `0`
- **No duplicate matched IDs within single list**: `True`
- **Duplicate matched ID list violations**: `0`
- **All matched IDs exist in S2 or S3**: `True`
- **Sample invalid matched IDs**: `[]`

## 9. Match Source Breakdown (S2 vs S3)

| Category | Count | Percentage (%) |
| --- | --- | --- |
| Both Source 2 & Source 3 | 1776047 | 80.4799 |
| Source 3 Only | 164498 | 7.4541 |
| Source 2 Only | 143029 | 6.4812 |
| No Matches (Singletons) | 123247 | 5.5848 |


## 10. Representative Ground-Truth Positive Pairs (Inspected)

| Source 1 ID | Source 1 Name | Source 1 Address | Matched Source | Matched ID | Matched Name | Matched Address | Name Exact Match |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S1-523975207 | VL Sprott | AL, Unit 3rd Phone, Dora, 125 Mountain View Lane | Source 3 | S3-804250913 | VL SPROTT | ##125 Mountain View Lane, # 3rd Phone, Dora, Alabama | True |


## 11. Representative Difficult Negative Pairs

| Source 1 ID | Source 1 Name | Negative ID | Negative Name | Negative Country | Shared Tokens |
| --- | --- | --- | --- | --- | --- |
| S1-925783039 | Orelee's Barbershop | S2-414418413 | CANDY'S-SERVICES | US | s |
| S1-925783039 | Orelee's Barbershop | S2-415733261 | Sweet Barbershop (Co) | US | barbershop |
| S1-925783039 | Orelee's Barbershop | S2-769808418 | Villafana's Strategic | US | s |
| S1-925783039 | Orelee's Barbershop | S2-34975790 | Fanny's Manufacturing | US | s |
| S1-925783039 | Orelee's Barbershop | S2-871017791 | Mayers's Inc Towing | US | s |
| S1-925783039 | Orelee's Barbershop | S2-41847464 | Donnelly's Buasl Manufacturing LLC | US | s |
| S1-925783039 | Orelee's Barbershop | S2-175566180 | catarina's security services | US | s |
| S1-925783039 | Orelee's Barbershop | S2-241291924 | CHENG'S  CONSULTING ACE | US | s |
| S1-925783039 | Orelee's Barbershop | S2-64137184 | Meade, J0annes S., P.A. Care | US | s |
| S1-925783039 | Orelee's Barbershop | S2-402804051 | Leonard's Brokerage LLC | US | s |
| S1-925783039 | Orelee's Barbershop | S2-327203402 | ONDREA'S-TOWING ALLIED | US | s |
| S1-925783039 | Orelee's Barbershop | S2-948826263 | KANE, AMIE F., D.D.S., INC | US | s |
| S1-925783039 | Orelee's Barbershop | S2-650540638 | Wendie's Wendie's Security LLC | US | s |
| S1-925783039 | Orelee's Barbershop | S2-162888799 | Coriss's Massage | US | s |
| S1-925783039 | Orelee's Barbershop | S2-383822500 | Advanced Avrit's LLC Wireless | US | s |
| S1-925783039 | Orelee's Barbershop | S2-48939070 | TRAN'S TOTAL MOTORS | US | s |
| S1-925783039 | Orelee's Barbershop | S2-655596796 | NOEMI'S MORTGAGE  INC | US | s |
| S1-925783039 | Orelee's Barbershop | S2-921776853 | ... Castanon's Barmt Eye Care Partners | US | s |
| S1-925783039 | Orelee's Barbershop | S2-209019917 | MARIAM'S OPTIMAL [CLEANING] | US | s |
| S1-925783039 | Orelee's Barbershop | S2-54521194 | Nakajima'S Manufacturing LP | US | s |


## 12. Candidate / Blocking Baseline Diagnostics

### `train_source1` Diagnostics
- **Unique Names**: 1,539,229 / 2,206,821 (69.75%)
- **Unique Addresses**: 2,130,606 / 2,206,821 (96.55%)
- **Top 15 Name Tokens**: `[('limited', 11921), ('private', 9890), ('llc', 8078), ('inc', 5515), ('ltd', 3366), ('pvt', 2717), ('india', 1398), ('and', 1295), ('care', 1257), ('associates', 998), ('of', 990), ('llp', 936), ('group', 889), ('center', 807), ('partners', 799)]`
- **Top 15 Address Tokens**: `[('road', 10739), ('no', 10032), ('delhi', 7591), ('street', 6967), ('drive', 4974), ('unit', 4654), ('avenue', 4351), ('maharashtra', 4325), ('nagar', 3746), ('floor', 3675), ('city', 3303), ('west', 3073), ('tx', 2985), ('mumbai', 2829), ('new', 2734)]`

### `train_source2` Diagnostics
- **Unique Names**: 4,402,008 / 5,034,616 (87.43%)
- **Unique Addresses**: 4,337,261 / 5,034,616 (89.14%)
- **Top 15 Name Tokens**: `[('private', 5312), ('limited', 5303), ('llc', 5279), ('inc', 3962), ('ltd', 3924), ('com', 2021), ('center', 1954), ('pvt', 1888), ('partners', 1541), ('services', 1507), ('co', 1444), ('corp', 1433), ('group', 1391), ('india', 1159), ('and', 1145)]`
- **Top 15 Address Tokens**: `[('no', 11328), ('road', 6812), ('delhi', 4856), ('street', 3521), ('st', 3380), ('maharashtra', 3341), ('rd', 3312), ('floor', 3248), ('city', 3089), ('tx', 3062), ('nagar', 3058), ('dr', 2923), ('ave', 2368), ('ny', 2341), ('pradesh', 2240)]`

### `train_source3` Diagnostics
- **Unique Names**: 4,651,608 / 5,285,603 (88.01%)
- **Unique Addresses**: 4,632,764 / 5,285,603 (90.67%)
- **Top 15 Name Tokens**: `[('limited', 6526), ('private', 6123), ('llc', 5383), ('ltd', 4114), ('inc', 3969), ('pvt', 2146), ('center', 2062), ('com', 2039), ('services', 1635), ('partners', 1611), ('group', 1384), ('co', 1324), ('corp', 1315), ('and', 1196), ('india', 1115)]`
- **Top 15 Address Tokens**: `[('no', 10703), ('road', 6198), ('new', 4693), ('delhi', 4442), ('city', 3541), ('street', 3379), ('north', 3296), ('rd', 3225), ('st', 3220), ('mh', 3188), ('nagar', 2865), ('texas', 2842), ('floor', 2762), ('dr', 2760), ('mumbai', 2717)]`

## 13. Observed Noise Patterns in the Dataset
1. **Legal Suffix Variations**: Extensive mix of `Inc`, `Incorporated`, `LLC`, `L.L.C.`, `Ltd`, `Limited`, `Pvt Ltd`, `Corp`, `Corporation`, `Co.`.
2. **Punctuation & Delimiters**: Frequent presence of periods, commas, slashes, ampersands vs 'and', hyphens, and inconsistent spaces.
3. **Address Component Permutations**: Street numbers, suite/unit numbers, postal codes, and city names appear in varied order or are omitted across sources.
4. **Case Inconsistency**: Mixed uppercase, lowercase, and title case across records.
5. **Typographical Errors**: Minor character insertions, omissions, and phonetic transliteration differences.

## 14. Initial Implications for Normalization (Observations)
- Text lowercasing, punctuation stripping, and whitespace collapse are essential baseline steps.
- Legal business entity suffix standardization (dictionary-based regex replacements) will significantly boost token overlap without losing entity identity.
- Street/address abbreviations (e.g. `St` -> `Street`, `Ave` -> `Avenue`, `Rd` -> `Road`, `Ste` -> `Suite`, `Bldg` -> `Building`) should be systematically normalized.
- Country strings should be strictly preserved and normalized for exact matching or blocking.

## 15. Initial Implications for Blocking & Candidate Generation (Observations)
- Given ~1.7M+ test entities, an exhaustive $O(N \times M)$ pairwise comparison is computationally impossible.
- Exact Country partitioning is a potent primary blocking boundary because entities virtually never cross national borders.
- Multi-key indexing combining normalized token prefixes, soundex/phonetic keys, or TF-IDF inverted indices will be crucial to maintain high recall (the recall ceiling) while keeping candidate pairs tractable.
- Singleton / no-match detection is paramount because a large proportion of Source 1 entities have zero true matches.
