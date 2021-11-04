# MMF-Data-Science-Project


<p>Sample Case Files (user_id = 'ALSAOZ1V546VT' in sample case):</p>

<ul>
<li>Create_itemCate.ipynb - Create item-category information csv (itemCate_2018.csv) from json data</li>
<li>DataPreProcess.ipynb - Preprocessing ratings data and create following data files:
                         <ul>
                         <li>inputItems.csv - Include all items the target user has rated and the corresponding ratings</li>
                         <li> userSubsetCate.csv - The dataframe of other users with common categories of item in 'inputItems.csv' and their corresponding mean rating of category-based</li>
                         </ul></li>
<li>recommend_system_code.ipynb - Actually recommendation system code for target user</li>
</ul>


<p>Test Case Files:</p>
<ul>
<li>DataCleanandPreProcess_FunctionVer.ipynb - Function version of data processing</li>

<li>test_cases.py - Test case, generate and validate test result</li>
