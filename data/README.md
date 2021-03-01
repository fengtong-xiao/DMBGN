## Introduction
A randomly desensitized sampled dataset from one of the large-scaled production dataset from from Lazada (Alibaba Group) is included. The dataset contains three dataframes corresponding users' voucher collection logs, related user behavior logs and related item features.

The dataset contains 3 dataframe stored in a pickle (.pkl) file which can be loaded with the following Python codes:

	import pickle
	file_path = './data/kdd_data.pkl'
	with open(file_path, "rb") as f:
	    log_df = pickle.load(f)
	    session_df = pickle.load(f)
	    item_df = pickle.load(f)
	log_df.shape, session_df.shape, item_df.shape

The description of each dataframe is as, note that all id features such as user_id, item_id, promotion_id, etc is hashed and desensitized, the timestamp is also hashed while the chronological order is preserved.

## 1. Voucher Collection Log (log_df) :
Users' voucher collection logs from Lazada voucher business which contains 62,068 records and 19 columns. Each record corresponding to a voucher being collected by a user with label = 0/1 represents whether the user redeems the voucher, the associated collection and redemption (if any) timestamp and user's profile feature is included as well. Data type and description of each column is as follows:


| Column Name                                        | Data Type | Description                                                                                                |
| -------------------------------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| session\_id                                        |  String   | A unique ID for each voucher collection record, the concatenation of user\_id and promotion\_id            |
| label                                              |  Integer  | Redemption label, 0 represents non-redeemed session, 1 represents redeemed session                         |
| user\_id                                           |  Integer  | Unique identification of a user                                                                            |
| promotion\_id                                      |  Integer  | Unique identification of a voucher (a.k.a promotion)                                                       |
| voucher\_min\_spend                                |  Integer  | The minimum spend amount of a voucher (in local currency)                                                  |
| voucher\_discount                                  |  Integer  | The discount amount of a voucher (in local currency)                                                       |
| voucher\_collect\_time                             |  Integer  | Timestamp of voucher being collected                                                                       |
| voucher\_redeem\_time                              |  Integer  | Timestamp of voucher being redeemed if label is 1, otherwise meaningless                                   |
| campaign\_name                                     |  String   | Campaign name of a voucher distribution activity, vouchers under the same campaign might affect each other |
| user\_age\_level                                   |  String   | User's predicted age level, from 0 to 8                                                                    |
| user\_gender                                       |  String   | User's predicted gender, '0' or '1' or 'default' (means not available)                                     |
| user\_purchase\_level                              |  String   | User's predicted purchase level, from 0 to 10                                                              |
| user\_trd\_\_orders\_cnt\_hist                     |  Double   | User's number of historical number of purchase                                                             |
| user\_trd\_\_orders\_cnt\_platform\_discount\_hist |  Double   | User's number of historical number of purchase using any voucher                                           |
| user\_trd\_\_actual\_gmv\_usd\_hist                |  Double   | Sum of GMV from user's historical order                                                                    |
| user\_trd\_\_max\_gmv\_usd\_hist                   |  Double   | Maximum GMV from all user's historical order                                                               |
| user\_trd\_\_avg\_gmv\_usd\_hist                   |  Double   | Average GMV from all user's historical order                                                               |
| user\_trd\_\_min\_gmv\_usd\_hist                   |  Double   | Minimum GMV from all user's historical order                                                               |
| dtype                                              |  String   | Represents training or testing data, 'train' or 'test'                                                     |
| rk                                                 |  Integer  | Chronological order of collection timestamp of each user                                                   |




## 2. User Behavior Logs Related to Voucher Collection (session\_df):
User's behavior logs (add-to-cart, order) happening both before and after the voucher collection, contains 1,118,593 recoreds and 14 columns. Each record contains a item\_id associated to a voucher collection session and the corresponding bahevior action\_type and timestamp. Data type and description of each column is as follows:

| Column Name               | Data Type | Description                                                                                                |
| ------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| label                     |  Integer  | Redemption label, 0 represents non-redeemed session, 1 represents redeemed session                         |
| session\_id               |  String   | A unique ID for each voucher collection record, the concatenation of user\\_id and promotion\\_id            |
| promotion\_id             |  Integer  | Unique identification of a voucher (a.k.a promotion)                                                       |
| voucher\_min\_spend       |  Integer  | The minimum spend amount of a voucher (in local currency)                                                  |
| voucher\_discount\_amount |  Integer  | The discount amount of a voucher (in local currency)                                                       |
| voucher\_collect\_time    |  Integer  | Timestamp of voucher being collected                                                                       |
| item\_id                  |  Integer  | Unique identification of an item                                                                           |
| action\_type              |  String   | User's behavior action type on this item from this session, 'cart' means add-to-cart, 'ord' means purchase |
| type                      |  String   | Represents whether the action is happened before ('bef') or after ('aft') the voucher collection           | 
| rk                        |  Integer  | Chronological order of behavior timestamp of each item from a user                                         |
| action\_time              |  Integer  | Timestamp of item being add-to-cart or order                                                               |
| item\_category\_id        |  Integer  | A unique ID for item's category                                                                            |
| item\_brand\_id           |  Integer  | A unique ID for item's brand                                                                               |
| item\_price\_level        |  Double   | Item's price level, from 1 to 7                                                                            |

## 3. Item Features (item\_df):
Item feature and pre-trained embedding (Section 3.4.2 in the submitted paper) of each associated item\_id in session\_df. Data type and description of each column is as follows:

| Column Name               | Data Type | Description                                                                                                |
| ------------------------- | --------- | ---------------------------------------------------------------------------------------------------------- |
| item\_id                  |  Integer  | Unique identification of an item                                                                           |
| atc_emb                   |  String   | Pretrained item embedding using all add-to-cart behaviors, dimension is 16, with ' ' as splitter           |
| ord_emb                   |  String   | Pretrained item embedding using all order behaviors, dimension is 16, with ' ' as splitter                 |
| brand\_id                 |  Integer  | A unique ID for item's brand                                                                               |
| category\_id              |  Integer  | A unique ID for item's category                                                                            |
| price\_level              |  Double   | Item's price level, from 1 to 7                                                                            |
