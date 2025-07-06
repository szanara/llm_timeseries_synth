import cohere
import numpy as np
import pandas as pd
import re
import glob
import time
from tqdm import tqdm 

key = 

def get_array(message):
    try:
        match = re.search(r'array\(\[(.*?)\]\)', gen, re.DOTALL)
        if match:
            array_str = match.group(1)
            return  array_str 
        else:
           raise ValueError("Padrão não encontrado no 1.")
    except (ValueError, AttributeError) as e:
        print(f'Erro: {e}')
        # Caso o padrão não tenha sido encontrado, tenta um padrão alternativo
        try:
            # Ajusta a expressão regular para possíveis variações
            padrao = re.compile(r'\[\[.*?\]\]', re.DOTALL)
            resultado = padrao.search(gen)
            if resultado:
                array_str = resultado.group(0)
                return  array_str 
            else:
               raise ValueError("Padrão não encontrado no 2.")
        except (ValueError, AttributeError) as e:
            print(f'Erro: {e}')
            return None


def final_array(message):
    array_str = get_array(message)
    if array_str is not None:
        data_str = array_str.replace('[', '').replace(']', '')
        array = np.fromstring(data_str, sep=',')
        if array.size % 3 != 0:
            raise ValueError("The array size is not divisible by 3, cannot reshape.")
        
        array = array.reshape(-1, 3)
        return array
    else: 
        return None

for i in range(3,5):
    for j in tqdm(range(12)):        
        with open(f'/home/cz/mds3/COHERE/real/MHEALTH/cat{j}-fold{i}.txt', 'r') as file:
            data = file.read()
        total_current_generated = len(glob.glob(f'/home/cz/mds3/COHERE/generated/mhealth/cat{j}-fold{i}-*.npy'))
        while total_current_generated < 30:
            co = cohere.Client(api_key=key)
            
            response = co.chat(
              model="command-r-plus",
              message=f"Generate ONE sample with shape (50,3) in the same distribution as the data I provided for you.I want the SAMPLE. Data:{data} "
            )
            gen = response.text
            print(f'FOLD{i}-cat{j}')
            
            try:
                array = final_array(gen)
                if array is not None:  # Ensure the array was successfully created
                    amount = total_current_generated + 1
                    file_path = f'/home/cz/mds3/COHERE/generated/mhealth/cat{j}-fold{i}-{amount}.npy'
                    np.save(file_path, array)
                    print(array.shape)
                else:
                    print("Array is None, not saving.")
            except ValueError as e:
                print(f"An error occurred: {e}")
            finally:
                total_current_generated = len(glob.glob(f'/home/cz/mds3/COHERE/generated/mhealth/cat{j}-fold{i}-*.npy'))
                time.sleep(5)




