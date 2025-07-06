import glob
import os
import os
import json
import google.generativeai as genai
import typing_extensions as typing
import numpy as np
import time
import fire

class My2DArray(typing.TypedDict):
    data: list[list[float]]

def run(cat_id, fold_id):
    print(f"Gerando para categoria {cat_id} e fold {fold_id}")
    dataset = "MHEALTH"
    filename = f"cat{cat_id}-fold{fold_id}.txt"
    file_path = f"{dataset}/data/{filename}"

    generate_folder_structure = f"{dataset}/generated/cat{cat_id}-fold{fold_id}-*.npy"

    arq = open(file_path, "r").read().replace("\n", "")

    prompt = f"""Generate one new vector sample with shape (50,3) in the same distribution as the data that I provided for you. 
    Data:
    {arq}
    """

    while len(glob.glob(generate_folder_structure)) < 30:
        try:
            all_already_generated = glob.glob(generate_folder_structure)

            key = 
            key =

            genai.configure(api_key=key)

            model = genai.GenerativeModel("gemini-1.5-flash-latest")
            result = model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    response_mime_type="application/json", response_schema=My2DArray
                )
            )

            v = result.to_dict()["candidates"]
            data = v[0]["content"]["parts"][0]["text"]
            vector = json.loads(data.strip())["data"]
            vector = np.asarray(vector)
            generated_idx = len(all_already_generated) + 1
            path2save = f"{dataset}/generated/cat{cat_id}-fold{fold_id}-{generated_idx}.npy"
            np.save(path2save, vector)
        except Exception as e:
            print("Não foi possível gerar nesse momento devido a este erro:", e)

        time.sleep(5)

if __name__ == "__main__":
    fire.Fire(run)
