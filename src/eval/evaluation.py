import math
import torch
from tqdm import tqdm
from collections import defaultdict
from src.utils.utils  import calculate_mean
from torch.utils.data import DataLoader


def evaluate_inidiv_dataset_LM(datasets, data_collator, batch_size, accelerator, model ,task="LM"):
    """
    Evaluate individual lanaguages
    """
    bpc_dictionary = {}
    loss_dictionary = {}
    stats_agg = defaultdict(list)
    for i in datasets:
        dataset = datasets[i]
        script_id = data_collator.tokenizer.encode(i,  add_special_tokens=False)[0]
        dataloader = DataLoader(dataset,
                                collate_fn=data_collator,
                                batch_size=batch_size,
                                shuffle=False)
        dataloader = accelerator.prepare(dataloader)
        count = 0
        losses1 = []
        losses2 = []
        for step, batch in enumerate(tqdm(dataloader, desc=f'evaluating {i} language...')):
            with torch.no_grad():
                seq_loss, seq_loss2,  stats, aux_loss, _ = model(batch, task=task, script_id=script_id) 
                count += 1
            losses1.append(accelerator.gather_for_metrics(seq_loss.repeat(batch_size)))
            losses2.append(accelerator.gather_for_metrics(seq_loss2.repeat(batch_size)))
            for k, v in stats.items():
                stats_agg[f"{i}_{k}"].append(v)

        losses1 = torch.cat(losses1)
        losses2 = torch.cat(losses2)
        try:
            eval_loss1 = torch.mean(losses1)
            eval_bpb1 = eval_loss1 / math.log(2)
        except OverflowError:
            eval_bpb1 = float("inf")
        try:
            eval_loss2 = torch.mean(losses2)
            eval_bpb2 = eval_loss2 / math.log(2)
        except OverflowError:
            eval_bpb2 = float("inf")

        bpc_dictionary[f"{i}_eval_bpb1"] = eval_bpb1.item()
        bpc_dictionary[f"{i}_eval_loss1"] = eval_loss1.item()
        bpc_dictionary[f"{i}_eval_bpb2"] = eval_bpb2.item()
        bpc_dictionary[f"{i}_eval_loss2"] = eval_loss2.item()

        print(f"Finished evaluating {i} language")
    stats_mean_dict = calculate_mean(stats_agg)
    bpc_dictionary.update(stats_mean_dict)
    print(bpc_dictionary)
    return bpc_dictionary, loss_dictionary

    
