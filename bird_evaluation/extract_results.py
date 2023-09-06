import json

bird_dev_set = json.load(open("./data/sft_eval_bird_text2sql.json"))
pred_sqls = [line.strip() for line in open("./pred_sqls.txt").readlines()]

print(len(bird_dev_set))
print(len(pred_sqls))

results = []
for data, pred_sql in zip(bird_dev_set, pred_sqls):
    results.append([pred_sql, data["db_id"]])

results_dict = dict()

for idx, result in enumerate(results):
    results_dict[idx] = result[0] + "\t----- bird -----\t" + result[1]

with open("./bird_evaluation/predict_dev.json", "w") as f:
    f.write(json.dumps(results_dict, indent=4))