import json
import os
import argparse

def split(txt_content):
    # input: txt
    # return: split by \n\n, then start, end position
    txt_ls = txt_content.split('\n\n')
    start_str = 0
    start_ls = []
    end_ls = []
    for i in range(len(txt_ls)):
        start_ls.append(start_str)
        end = start_str+len(txt_ls[i])
        end_ls.append(end)
        start_str = end+2
    return txt_ls, start_ls, end_ls

def judge_in_sentence(txt_ls, start_ls, end_ls, label_ls):
    res = [[] for j in range(len(txt_ls))]
    for i in range(len(txt_ls)):
        #print ("<<< i", i )
        for j in range(len(label_ls)):
            if label_ls[j]['start_offset']>=start_ls[i] and label_ls[j]['end_offset']<=end_ls[i]:
                #print ("label_ls[j]",label_ls[j])
                #print ("start_ls[i]: {} end_ls[i]: {} ".format(start_ls[i],end_ls[i]))
                #update the index postions
                #label_ls[j]['start_offset'] = label_ls[j]['start_offset']-start_ls[i]
                #label_ls[j]['end_offset'] = label_ls[j]['end_offset']-start_ls[i]
                res[i].append({'id': label_ls[j]['id'],
                               'label':label_ls[j]['label'],
                                'start_offset':label_ls[j]['start_offset']-start_ls[i],
                                'end_offset': label_ls[j]['end_offset']-start_ls[i]})
            else:
                continue
    return res

def judge_relations(res_entity,label_ls):
    res = [[] for j in range(len(res_entity))]
    for i in range(len(res_entity)):
        entity = res_entity[i]
        label_all = [i['id'] for i in entity]
        for j in label_ls:
            if j['from_id'] in label_all and j['to_id'] in label_all:
                res[i].append(j)
            else:
                continue
    return res

def write_back(raw_examples):
    res = []
    idx = 0
    for i in range(len(raw_examples)):
        x1 = json.loads(raw_examples[i])
        txt_ls, start_ls, end_ls = split(x1['text'])
        res_entity = judge_in_sentence(txt_ls, start_ls, end_ls, x1['entities'])
        res_relation = judge_relations(res_entity, x1['relations'])
        for j in range(len(txt_ls)):
            #only save content length larger than 10
            #only save the texts that have labels
            if len(txt_ls[j])>10 and len(res_entity[j])>0:
                dict_res = {'id': idx,
                            'text': txt_ls[j],
                            'entities': res_entity[j],
                            'relations': res_relation[j]}
                res.append(dict_res)
                idx = idx + 1
    return res

def save_single_file(input_path, output_folder):
    with open(input_path, "r", encoding="utf-8") as f:
        raw_examples = f.readlines()

    res = write_back(raw_examples)
    # write back
    with open(os.path.join(output_folder, input_path.split('/')[-1]), 'w', encoding="utf-8") as json_file:
        for i in res:
            json_file.write(json.dumps(i, ensure_ascii=False) + "\n")

def main(mode, input_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if mode=='single_file':
        save_single_file(input_path, output_folder)
    elif mode=='folder':
        files = os.listdir(input_path)
        for i in files:
            if i!='.ipynb_checkpoints':
                save_single_file(os.path.join(input_path,i), output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training pipeline')
    parser.add_argument('--mode', type=str, default='single_file',
                        help="process single_file or folder")
    parser.add_argument('--input_path', type=str, default='./', help="input file or input folder path")
    parser.add_argument('--output_folder', type=str, default='./', help="output folder path")

    args = parser.parse_args()

    main(args.mode, args.input_path, args.output_folder)