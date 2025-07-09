import configparser
import dataclasses
import json
import os
import re
from typing import List

import retry
from openai import OpenAI
from google import genai
from tqdm import tqdm

from arxiv_scraper import Paper
from arxiv_scraper import EnhancedJSONEncoder


def filter_by_author(all_authors, papers, author_targets, config):
    # filter and parse the papers
    selected_papers = {}  # pass to output
    all_papers = {}  # dict for later filtering
    sort_dict = {}  # dict storing key and score

    # author based selection
    for paper in papers:
        all_papers[paper.arxiv_id] = paper
        for author in paper.authors:
            if author in all_authors:
                for alias in all_authors[author]:
                    if alias["authorId"] in author_targets:
                        selected_papers[paper.arxiv_id] = {
                            **dataclasses.asdict(paper),
                            **{"COMMENT": "Author match"},
                        }
                        sort_dict[paper.arxiv_id] = float(
                            config["SELECTION"]["author_match_score"]
                        )
                        break
    return selected_papers, all_papers, sort_dict


def filter_papers_by_hindex(all_authors, papers, config):
    # filters papers by checking to see if there's at least one author with > hcutoff hindex
    paper_list = []
    for paper in papers:
        max_h = 0
        for author in paper.authors:
            if author in all_authors:
                max_h = max(
                    max_h, max([alias["hIndex"] for alias in all_authors[author]])
                )
        if max_h >= float(config["FILTERING"]["hcutoff"]):
            paper_list.append(paper)
    return paper_list


def calc_token_usage(usage):
    """
    返回输入和输出的token计数，处理不同客户端的usage对象
    """
    if hasattr(usage, 'prompt_tokens'):
        # OpenAI 格式
        return {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.prompt_tokens + usage.completion_tokens
        }
    else:
        # Gemini 格式，返回默认值
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


@retry.retry(tries=3, delay=2)
def call_chatgpt(full_prompt, ai_client, model):
    if isinstance(ai_client, OpenAI):
        # OpenAI 客户端
        return ai_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": full_prompt}],
            temperature=0.0,
            seed=0,
        )
    else:
        # Gemini 客户端
        return ai_client.models.generate_content(
            model=model,
            contents=full_prompt
        )


def run_and_parse_chatgpt(full_prompt, ai_client, config):
    # 运行AI客户端并解析JSON响应
    completion = call_chatgpt(full_prompt, ai_client, config["SELECTION"]["model"])
    
    if isinstance(ai_client, OpenAI):
        # OpenAI 响应格式
        out_text = completion.choices[0].message.content
        usage = completion.usage
    else:
        # Gemini 响应格式
        out_text = completion.text
        usage = None
    
    out_text = re.sub("```jsonl\n", "", out_text)
    out_text = re.sub("```", "", out_text)
    out_text = re.sub(r"\n+", "\n", out_text)
    out_text = re.sub("},", "}", out_text).strip()
    
    # 逐行解析JSON
    json_dicts = []
    for line in out_text.split("\n"):
        try:
            json_dicts.append(json.loads(line))
        except Exception as ex:
            if config["OUTPUT"].getboolean("debug_messages"):
                print("Exception happened " + str(ex))
                print("Failed to parse LM output as json")
                print(out_text)
                print("RAW output")
                if isinstance(ai_client, OpenAI):
                    print(completion.choices[0].message.content)
                else:
                    print(completion.text)
            continue
    return json_dicts, calc_token_usage(usage) if usage else {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def paper_to_string(paper_entry: Paper) -> str:
    # renders each paper into a string to be processed by GPT
    new_str = (
        "ArXiv ID: "
        + paper_entry.arxiv_id
        + "\n"
        + "Title: "
        + paper_entry.title
        + "\n"
        + "Authors: "
        + " and ".join(paper_entry.authors)
        + "\n"
        + "Abstract: "
        + paper_entry.abstract[:4000]
    )
    return new_str


def batched(items, batch_size):
    # takes a list and returns a list of list with batch_size
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def limit_papers_by_score(papers_dict, sort_dict, max_papers=100):
    """
    Limit the number of papers to max_papers by keeping only the highest-scoring ones.
    This prevents the GitHub.io page from exceeding 400KB and failing to render.

    Args:
        papers_dict: Dictionary mapping paper IDs to paper data
        sort_dict: Dictionary mapping paper IDs to scores
        max_papers: Maximum number of papers to keep

    Returns:
        Dictionary with at most max_papers papers, keeping the highest-scoring ones
    """
    if len(papers_dict) <= max_papers:
        return papers_dict

    # Sort papers by score in descending order
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())

    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    sorted_indices = argsort(values)[::-1]  # Descending order
    top_keys = [keys[idx] for idx in sorted_indices[:max_papers]]

    # Create a new dictionary with only the top papers
    limited_papers = {key: papers_dict[key] for key in top_keys}

    return limited_papers


def filter_papers_by_title(
    papers, config, ai_client, base_prompt, criterion
) -> List[Paper]:
    filter_postfix = 'Identify any papers that are absolutely and completely irrelavent to the criteria, and you are absolutely sure your friend will not enjoy, formatted as a list of arxiv ids like ["ID1", "ID2", "ID3"..]. Be extremely cautious, and if you are unsure at all, do not add a paper in this list. You will check it in detail later.\n Directly respond with the list, do not add ANY extra text before or after the list. Even if every paper seems irrelevant, please keep at least TWO papers'
    batches_of_papers = batched(papers, 20)
    final_list = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for batch in batches_of_papers:
        papers_string = "".join([paper_to_titles(paper) for paper in batch])
        full_prompt = (
            base_prompt + "\n " + criterion + "\n" + papers_string + filter_postfix
        )
        model = config["SELECTION"]["model"]
        completion = call_chatgpt(full_prompt, ai_client, model)
        
        if isinstance(ai_client, OpenAI):
            out_text = completion.choices[0].message.content
            token_usage = calc_token_usage(completion.usage)
        else:
            out_text = completion.text
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        
        total_prompt_tokens += token_usage["prompt_tokens"]
        total_completion_tokens += token_usage["completion_tokens"]
        
        try:
            filtered_set = set(json.loads(out_text))
            for paper in batch:
                if paper.arxiv_id not in filtered_set:
                    final_list.append(paper)
                else:
                    print("Filtered out paper " + paper.arxiv_id)
        except Exception as ex:
            print("Exception happened " + str(ex))
            print("Failed to parse LM output as list " + out_text)
            print(completion)
            continue
    return final_list, {"prompt_tokens": total_prompt_tokens, "completion_tokens": total_completion_tokens, "total_tokens": total_prompt_tokens + total_completion_tokens}


def paper_to_titles(paper_entry: Paper) -> str:
    return "ArXiv ID: " + paper_entry.arxiv_id + " Title: " + paper_entry.title + "\n"


def run_on_batch(
    paper_batch, base_prompt, criterion, postfix_prompt, ai_client, config
):
    batch_str = [paper_to_string(paper) for paper in paper_batch]
    full_prompt = "\n".join(
        [
            base_prompt,
            criterion + "\n",
            "\n\n".join(batch_str) + "\n",
            postfix_prompt,
        ]
    )
    json_dicts, token_usage = run_and_parse_chatgpt(full_prompt, ai_client, config)
    return json_dicts, token_usage


def filter_by_gpt(
    all_authors, papers, config, ai_client, all_papers, selected_papers, sort_dict
):
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    if config["SELECTION"].getboolean("run_openai"):
        # filter first by hindex of authors to reduce costs.
        paper_list = filter_papers_by_hindex(all_authors, papers, config)
        if len(paper_list) < 5:
            print(f'Only {len(paper_list)} papers left, something might be wrong with the hindex filtering, use all papers instead.')
            paper_list = papers[:]
        if config["OUTPUT"].getboolean("debug_messages"):
            print(str(len(paper_list)) + " papers after hindex filtering")
        paper_list, token_usage = filter_papers_by_title(
            paper_list, config, ai_client, base_prompt, criterion
        )
        if config["OUTPUT"].getboolean("debug_messages"):
            print(
                str(len(paper_list))
                + f" papers after title filtering with token usage: {token_usage['prompt_tokens']} prompt tokens, {token_usage['completion_tokens']} completion tokens"
            )
        total_prompt_tokens += token_usage["prompt_tokens"]
        total_completion_tokens += token_usage["completion_tokens"]

        # batch the remaining papers and invoke GPT
        batch_of_papers = batched(paper_list, int(config["SELECTION"]["batch_size"]))
        scored_batches = []
        for batch in tqdm(batch_of_papers):
            scored_in_batch = []
            json_dicts, token_usage = run_on_batch(
                batch, base_prompt, criterion, postfix_prompt, ai_client, config
            )
            total_prompt_tokens += token_usage["prompt_tokens"]
            total_completion_tokens += token_usage["completion_tokens"]
            for jdict in json_dicts:
                if (
                    int(jdict["RELEVANCE"])
                    >= int(config["FILTERING"]["relevance_cutoff"])
                    and jdict["NOVELTY"] >= int(config["FILTERING"]["novelty_cutoff"])
                    and jdict["ARXIVID"] in all_papers
                ):
                    selected_papers[jdict["ARXIVID"]] = {
                        **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                        **jdict,
                    }
                    sort_dict[jdict["ARXIVID"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]
                scored_in_batch.append(
                    {
                        **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                        **jdict,
                    }
                )
            scored_batches.append(scored_in_batch)
        # Limit the number of papers to 100 to prevent GitHub.io page from exceeding 400KB
        limited_papers = limit_papers_by_score(selected_papers, sort_dict)
        if len(limited_papers) != len(selected_papers):
            # Update the selected_papers dictionary with the limited set
            selected_papers.clear()
            selected_papers.update(limited_papers)
        else:
            # If lengths are the same, limit_papers_by_score likely returned the original selected_papers
            # or a filtered version with the same number of elements. No need to clear/update.
            pass

        if config["OUTPUT"].getboolean("dump_debug_file"):
            with open(
                config["OUTPUT"]["output_path"] + "gpt_paper_batches.debug.json", "w"
            ) as outfile:
                json.dump(scored_batches, outfile, cls=EnhancedJSONEncoder, indent=4)
        if config["OUTPUT"].getboolean("debug_messages"):
            print(f"Total token usage: {total_prompt_tokens} prompt tokens, {total_completion_tokens} completion tokens")
            print(f"Limited papers from {len(sort_dict)} to {len(selected_papers)} to prevent GitHub.io page from exceeding 400KB")


if __name__ == "__main__":
    config = configparser.ConfigParser()
    config.read("configs/config.ini")
    
    # 使用环境变量确定客户端类型
    CLIENT = os.environ.get("CLIENT", "openai").lower()
    
    if CLIENT == "gemini":
        GEMINI_KEY = os.environ.get("GEMINI_KEY")
        if GEMINI_KEY is None:
            raise ValueError(
                "Gemini key is not set - please set GEMINI_KEY to your Gemini key"
            )
        ai_client = genai.Client(api_key=GEMINI_KEY)
    else:
        # 回退到 keys.ini 文件以兼容旧版本
        keyconfig = configparser.ConfigParser()
        keyconfig.read("configs/keys.ini")
        ai_client = OpenAI(api_key=keyconfig["KEYS"]["openai"], base_url=keyconfig["KEYS"]["openai_base_url"])
    # deal with config parsing
    with open("configs/base_prompt.txt", "r") as f:
        base_prompt = f.read()
    with open("configs/paper_topics.txt", "r") as f:
        criterion = f.read()
    with open("configs/postfix_prompt.txt", "r") as f:
        postfix_prompt = f.read()
    # loads papers from 'in/debug_papers.json' and filters them
    with open("in/debug_papers.json", "r") as f:
        # with open("in/gpt_paper_batches.debug-11-10.json", "r") as f:
        paper_list_in_dict = json.load(f)
    papers = [
        [
            Paper(
                arxiv_id=paper["arxiv_id"],
                authors=paper["authors"],
                title=paper["title"],
                abstract=paper["abstract"],
            )
            for paper in batch
        ]
        for batch in paper_list_in_dict
    ]
    all_papers = {}
    paper_outputs = {}
    sort_dict = {}
    total_prompt_tokens = 0
    total_completion_tokens = 0
    for batch in tqdm(papers):
        json_dicts, token_usage = run_on_batch(
            batch, base_prompt, criterion, postfix_prompt, ai_client, config
        )
        total_prompt_tokens += token_usage["prompt_tokens"]
        total_completion_tokens += token_usage["completion_tokens"]
        for paper in batch:
            all_papers[paper.arxiv_id] = paper
        for jdict in json_dicts:
            paper_outputs[jdict["ARXIVID"]] = {
                **dataclasses.asdict(all_papers[jdict["ARXIVID"]]),
                **jdict,
            }
            sort_dict[jdict["ARXIVID"]] = jdict["RELEVANCE"] + jdict["NOVELTY"]

        # sort the papers by relevance and novelty
    print(f"Total token usage: {total_prompt_tokens} prompt tokens, {total_completion_tokens} completion tokens")
    keys = list(sort_dict.keys())
    values = list(sort_dict.values())

    def argsort(seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    sorted_keys = [keys[idx] for idx in argsort(values)[::-1]]
    selected_papers = {key: paper_outputs[key] for key in sorted_keys}

    # Limit the number of papers to 100 to prevent GitHub.io page from exceeding 400KB
    selected_papers = limit_papers_by_score(selected_papers, sort_dict)
    print(f"Limited papers from {len(sort_dict)} to {len(selected_papers)} to prevent GitHub.io page from exceeding 400KB")

    with open(
        config["OUTPUT"]["output_path"] + "filter_paper_test.debug.json", "w"
    ) as outfile:
        json.dump(selected_papers, outfile, cls=EnhancedJSONEncoder, indent=4)
