UML Code Analysis Project
Main Prepared Datasets in Excel Files
1. file test-dataset1.xlsx: Contains all generated codes that are used by the programs listed in rows 1, 2, and 3.
2. file test-dataset2.xlsx: Contains all generated codes used by the program in row 4.
3. Error.xls: Contains a review of the number of prompts for which their UML code image has not been rendered.
4. Total.xlsx: Contains all analyses.
The main program folder is: F:\uml\mohammadT
Important Notes on This Folder (Files and Their Purposes)
1. Scoring prompts using huggingface.py: Used for scoring and assessing prompt complexity; its database is test-dataset2.xlsx.
2. Huggingfacs Standard prompt.py: Used for prompt standardization, using NLP to convert simple prompts to their revised forms. its database is test-dataset2.xlsx.
3. all-comparings2-2.py: Used for Syntetic evaluation; its database is test-dataset1.xlsx.
4. PlantUML_similarity_combined_v2.py: Used for similarity analysis (Cosine, TF_IDF, etc.); its database is test-dataset1.xlsx.
5. prompt and code correspondingwith huggingface.py: Determines the similarity between generated codes and prompt texts; its database is test-dataset2.xlsx.
6. Image plantuml render_excel_plantuml_to_jpg.py: Converts code to images. Place files under f:\uml\mohammadt (can be changed in the code). Reads from Excel, creates a branch for each column, and saves images per branch (e.g., each branch is a model like cld001). Ensure the PlantUML extension is installed in VS Code with relevant libraries.
7. selected prompt to assessing.py: For human evaluation of representative samples; finally, 9 prompts were selected.
8. MergeImage.py: Collects all images of each model for human-evaluated selected prompts, storing them in a folder named with the model and the prompt row name.
Special Notes
• F:\uml\mohammadT\Results contains output files from each Python script, and each review sheet is eventually transferred (with the same reviewed filename) to F:\uml\mohammadT\New_Results\resulttttt\Total.xls.
• Main Excel files used in analyses are located in F:\uml\mohammadT\New_Results\resulttttt.
Specific Output Files
1. LL_New Python_ALL_comparing_dataset_with_parameters1.xlsx: Output of all-comparings2-2.py for syntactic analyses.
2. ALL_New TF-IDF-similarity_output_similarity.xlsx: Output of PlantUML_similarity_combined_v2.py.
   a. ALL_New TF-IDF-similarity_statistics_output_similarity.xlsx: For similarity analysis.
3. ALL_New Prompt_huggingface_results1.xlsx: Output of prompt and code correspondingwith huggingface.py.
   a. ALL_New Prompt_metric_statistics_huggingface1.xlsx: For analyzing prompt-code similarity.
How to Work with the Files
1. Copy the desired UML code columns for each model into the first sheet of test-dataset1.xlsx.
2. Copy the prompt title column and desired UML code columns for each model into the first sheet of test-dataset2.xlsx.
all-comparings2.py is used for producing outputs and numeric analysis of UML code elements (Sytetic analysis) with Python commands; outputs as CSV.
• To use it, copy all UML code columns for all models (in separate columns) into the first sheet of test-dataset1.xlsx, save, and process.
• Copy and reverse-paste the output from the CSV and the created sheet into the master file (total.xlsx) for full analysis.
For similarity and TF_IDF analysis (Similarities analysis), use PlantUML_similarity_combined_v2.py:
• Copy-paste the columns for each model separately (as above) into test-dataset1.xlsx's first sheet and save.
• Each model must be saved and the script run separately, then paste both output files side-by-side in two sheets in total.xlsx.
Important: When running all-comparings2.py, copy all model columns at once. For the other scripts, do this for each model separately.
To find which program uses which input and output files, search for ".XLSX" and ".CSV" in the scripts.
After transferring data to total.xlsx, all three method outputs are available for further statistical analysis. The Python programs (PlantUML_similarity_combined_v2.py and All-comparings1.py) also produce a general statistical analysis output.
