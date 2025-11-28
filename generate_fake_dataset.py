import argparse
import random
import pandas as pd
import numpy as np
import json
import string


def generate_fake_dataset(
        num_rows,
        domains,
        domain_ratio,
        domain_field_config,
        fixed_user_item_k,
        sequence_length_config,
        scenario_config,
        value_range,
        concrete_range,
        ratio_noise,
        save_path
):
    app_id_map = {d: i for i, d in enumerate(domains)}

    rows = []

    # 固定字段
    fixed_fields = []
    for i in range(1, fixed_user_item_k + 1):
        fixed_fields.append(f"user_{i}")
    for i in range(1, fixed_user_item_k + 1):
        fixed_fields.append(f"item_{i}")

    # 域字段
    domain_continue_fields = []
    domain_concrete_fields = []
    sequence_fields = []

    for domain in domains:
        conf = domain_field_config[domain]

        for obj in ["user", "item"]:
            count = conf[obj]["continue"]
            for i in range(1, count + 1):
                domain_continue_fields.append(f"{domain}_{obj}_continue_{i}")

        for obj in ["user", "item"]:
            count = conf[obj]["concrete"]
            for i in range(1, count + 1):
                domain_concrete_fields.append(f"{domain}_{obj}_concrete_{i}")

        for obj in ["user", "item"]:
            seq_count = conf.get(f"sequence_{obj}", 0)
            for j in range(seq_count):
                sequence_fields.append(f"{domain}_{obj}_sequen{j + 1}")

    special_fields = ["domain_id", "userID", "itemID", "scenario_id", "is_click", "is_play", "is_pay"]

    # ---- 比例扰动 ----
    ratios = np.array([domain_ratio[d] for d in domains], dtype=float)
    noise = np.random.uniform(-ratio_noise, ratio_noise, size=len(domains))
    ratios = ratios + noise
    ratios = np.maximum(ratios, 0)

    if ratios.sum() == 0:
        ratios = np.ones_like(ratios) / len(ratios)
    else:
        ratios = ratios / ratios.sum()

    domain_choices = np.random.choice(domains, size=num_rows, p=ratios)

    # ---- 生成数据 ----
    for idx in range(num_rows):
        current_domain = domain_choices[idx]

        record = {}

        # app_id
        record["domain_id"] = app_id_map[current_domain]

        # 基本字段
        record["userID"] = random.randint(0, 99999)
        record["itemID"] = random.randint(0, 99999)

        # scenario
        max_scen = scenario_config[current_domain]
        record["scenario_id"] = random.randint(0, max_scen - 1)

        record["is_click"] = random.randint(0, 1)
        record["is_play"] = random.randint(0, 1)
        record["is_pay"] = random.randint(0, 1)
        record["image_features"] = [random.uniform(*(0, 255)) for _ in range(64)]
        # record["text_features"] = ''.join(random.choices(string.ascii_letters + string.digits, k=256))
        record["text_features"] = [random.uniform(*(0, 255)) for _ in range(64)]

        # 固定字段
        for f in fixed_fields:
            record[f] = random.uniform(*value_range)

        # 域字段
        for field in domain_continue_fields:
            domain = field.split("_", 1)[0]
            if domain == current_domain:
                record[field] = random.uniform(*value_range)
            else:
                record[field] = 0

        for field in domain_concrete_fields:
            domain = field.split("_", 1)[0]
            if domain == current_domain:
                record[field] = random.randint(int(concrete_range[0]), int(concrete_range[1]))
            else:
                record[field] = 0

        # 序列字段
        seq_len = sequence_length_config[current_domain]
        for field in sequence_fields:
            domain = field.split("_", 1)[0]
            if domain == current_domain:
                seq = [random.uniform(*value_range) for _ in range(seq_len)]
            else:
                seq = [0.0] * seq_len

            record[field] = json.dumps(seq)

        rows.append(record)

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False)
    print(f"✔ 已生成: {save_path}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--num_rows", type=int, default=200)
    parser.add_argument("--domains", type=str, default="game,video,music,theme,read")
    parser.add_argument("--domain_ratio", type=str,
                        default="game:0.05,video:0.7,music:0.1,theme:0.1,read:0.05")
    parser.add_argument("--domain_field_config", type=str, required=True)
    parser.add_argument("--sequence_length_config", type=str, required=True)
    parser.add_argument("--scenario_config", type=str, required=True)

    parser.add_argument("--fixed_user_item_k", type=int, default=3)
    parser.add_argument("--ratio_noise", type=float, default=0.05)
    parser.add_argument("--value_range", type=str, default="0,1")
    parser.add_argument("--concrete_range", type=str, default="0,99")
    parser.add_argument("--save_path", type=str, default="fake_dataset.csv")

    args = parser.parse_args()

    # 字符串解析
    domains = args.domains.split(",")

    domain_ratio = {
        kv.split(":")[0]: float(kv.split(":")[1])
        for kv in args.domain_ratio.split(",")
    }

    value_low, value_high = map(float, args.value_range.split(","))
    concrete_value_low, concrete_value_high = map(float, args.concrete_range.split(","))
    # JSON 方式传入复杂结构（最常见）
    domain_field_config = json.loads(args.domain_field_config)
    sequence_length_config = json.loads(args.sequence_length_config)
    scenario_config = json.loads(args.scenario_config)

    generate_fake_dataset(
        num_rows=args.num_rows,
        domains=domains,
        domain_ratio=domain_ratio,
        domain_field_config=domain_field_config,
        fixed_user_item_k=args.fixed_user_item_k,
        sequence_length_config=sequence_length_config,
        scenario_config=scenario_config,
        value_range=(value_low, value_high),
        concrete_range=(concrete_value_low, concrete_value_high),
        ratio_noise=args.ratio_noise,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
