**UML Code Analysis Project**

**Input datasets:**

1. file data/row_prompt.csv:    Contains all user stories
2. file data/test_dataset1.xlsx:     Contains all generated PlantUml codes.
3. file data/test_dataset2.xlsx:    Contains all generated PlantUml codes with Simple and Modified stories.

**Output datasets, .etc (Including all results from the Python Scripts):**

1. Analysis/Analysis.xlsx: Contains all Final analyses.
2. All output files of Programs are in /report
   
**Pythons code scripts:**

1.	Src/01_Scoring_prompts_using_huggingface.py: 
Used for assessing raw prompt complexity, it’s database is "data/row_promt.csv"
2. Src/02_Huggingfacs_Transformer_Standard_prompt.py: 
Used for prompt standardization, using NLP to convert simple prompts to their revised forms. it’s database is "data/row_promt.csv"
3. Src/03_Syntatic_Elemnt_Score.py: 
Used for Syntetic evaluation; "data/ test_dataset1.xlsx”.
4. Src/04_PlantUML_similarities.py: Used for similarity analysis (Cosine, TF_IDF, etc.); "data/test-dataset1.xlsx”.
5. Src/05_Symantic_prompt_and_code_coresponding_huggingface.py: 
Determines the similarity between generated codes and prompt texts; its database is test-dataset2.xlsx. Note: It must be run twice: once for the simple prompts and once for the modified prompts. Each time, the relevant sheets from the Excel file must be manually copied/pasted into the first sheet ("Modifiedprompt") according to the prompt type, and the output files for analysis should also be renamed accordingly. it’s database is "data// test_dataset2.xlsx"
6. Src/06_Image_plantuml_render_to_jpg.py:
Converts generated PlantUml codes to images in folder ”report/jpg “. Reads from Excel, creates a subfolder for each model, and saves their images in the subfolder (e.g., each subfolder is a model like cld001, Claud in tempreture 0.0). Ensure the PlantUML extension is installed in VS Code with relevant libraries.
7. Src/07_Select_optimal_prompts_for_human_assessing.py:
Due to the large volume of generated code, it prepares representative samples for human evaluation using statistical methods.
8. Src/08_Group_MergeImage_for_selected_prompt_Humanasessing.py: 
This program collects the drawn images from all models for the prompts selected for human evaluation and places them in the **Merg** folder.  
As a template, in the **ABC001** folder, we first create hypothetical images named after the selected prompt rows, and then we run the program.

**How to Work with the Files**

The commands inside the **Install.txt** file must be executed.
NOTE: Finally, all outputs of the mentioned programs have been reviewed in an Excel file for analysis, and these programs have only generated outputs that were later analyzed and reviewed, with the results visible in the **/Analysis** folder.
 
<img width="837" he
   ight="645" alt="image" src="https://github.com/user-attachments/assets/ea1fc66d-9227-460f-a442-19c924745590" />
