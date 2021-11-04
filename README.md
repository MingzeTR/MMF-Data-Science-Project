# MMF-Data-Science-Project

Sample Case Files:
Create_itemCate.ipynb - Create item-category information csv (itemCate_2018.csv) from json data
DataPreProcess.ipynb - Preprocessing ratings data and create:
                         inputItems.csv - Include all items the target user (user_id = 'ALSAOZ1V546VT' in sample case) has rated and the corresponding ratings
                         userSubsetCate.csv - The dataframe of other users with common categories of item in 'inputItems.csv' and their corresponding mean rating of category-based 
recommend_system_code.ipynb - Actually recommendation system code for target user (user_id = 'ALSAOZ1V546VT' in sample case)

Test Case Files:
DataCleanandPreProcess_FunctionVer.ipynb - Function version of data processing
test_cases.py - Test case and generate test result
