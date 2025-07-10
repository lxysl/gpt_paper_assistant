import json
from datetime import datetime
import re

link_prefix = 'user-content-'
topic_shift = 1000

def render_paper(paper_entry: dict, idx: int) -> str:
    """
    :param paper_entry: is a dict from a json. an example is
    {"paperId": "2754e70eaa0c2d40972c47c4c23210f0cece8bfc", "externalIds": {"ArXiv": "2310.16834", "CorpusId": 264451832}, "title": "Discrete Diffusion Language Modeling by Estimating the Ratios of the Data Distribution", "abstract": "Despite their groundbreaking performance for many generative modeling tasks, diffusion models have fallen short on discrete data domains such as natural language. Crucially, standard diffusion models rely on the well-established theory of score matching, but efforts to generalize this to discrete structures have not yielded the same empirical gains. In this work, we bridge this gap by proposing score entropy, a novel discrete score matching loss that is more stable than existing methods, forms an ELBO for maximum likelihood training, and can be efficiently optimized with a denoising variant. We scale our Score Entropy Discrete Diffusion models (SEDD) to the experimental setting of GPT-2, achieving highly competitive likelihoods while also introducing distinct algorithmic advantages. In particular, when comparing similarly sized SEDD and GPT-2 models, SEDD attains comparable perplexities (normally within $+10\\%$ of and sometimes outperforming the baseline). Furthermore, SEDD models learn a more faithful sequence distribution (around $4\\times$ better compared to GPT-2 models with ancestral sampling as measured by large models), can trade off compute for generation quality (needing only $16\\times$ fewer network evaluations to match GPT-2), and enables arbitrary infilling beyond the standard left to right prompting.", "year": 2023, "authors": [{"authorId": "2261494043", "name": "Aaron Lou"}, {"authorId": "83262128", "name": "Chenlin Meng"}, {"authorId": "2490652", "name": "Stefano Ermon"}], "ARXIVID": "2310.16834", "COMMENT": "The paper shows a significant advance in the performance of diffusion language models, directly meeting one of the criteria.", "RELEVANCE": 10, "NOVELTY": 8}, "2310.16779": {"paperId": "edc8953d559560d3237fc0b27175cdb1114c0ca5", "externalIds": {"ArXiv": "2310.16779", "CorpusId": 264451949}, "title": "Multi-scale Diffusion Denoised Smoothing", "abstract": "Along with recent diffusion models, randomized smoothing has become one of a few tangible approaches that offers adversarial robustness to models at scale, e.g., those of large pre-trained models. Specifically, one can perform randomized smoothing on any classifier via a simple\"denoise-and-classify\"pipeline, so-called denoised smoothing, given that an accurate denoiser is available - such as diffusion model. In this paper, we investigate the trade-off between accuracy and certified robustness of denoised smoothing: for example, we question on which representation of diffusion model would maximize the certified robustness of denoised smoothing. We consider a new objective that aims collective robustness of smoothed classifiers across multiple noise levels at a shared diffusion model, which also suggests a new way to compensate the cost of accuracy in randomized smoothing for its certified robustness. This objective motivates us to fine-tune diffusion model (a) to perform consistent denoising whenever the original image is recoverable, but (b) to generate rather diverse outputs otherwise. Our experiments show that this fine-tuning scheme of diffusion models combined with the multi-scale smoothing enables a strong certified robustness possible at highest noise level while maintaining the accuracy closer to non-smoothed classifiers.", "year": 2023, "authors": [{"authorId": "83125078", "name": "Jongheon Jeong"}, {"authorId": "2261688831", "name": "Jinwoo Shin"}], "ARXIVID": "2310.16779", "COMMENT": "The paper presents an advancement in the performance of diffusion models, specifically in the context of denoised smoothing.", "RELEVANCE": 9, "NOVELTY": 7, "TLDR": "xxx"}
    :return: a markdown formatted string showing the arxiv id, title, arxiv url, abstract, authors, score, comment and tldr (if those fields exist)
    """
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"]
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    arxiv_pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    # get the abstract
    abstract = paper_entry["abstract"]
    # get the authors
    authors = paper_entry["authors"]
    paper_string = f'<a id="paper-{idx}"></a>\n### {idx}. [{title}]({arxiv_url})\n'
    paper_string += f"**ArXiv:** {arxiv_id} [[page]({arxiv_url})] [[pdf]({arxiv_pdf_url})] [[kimi](https://papers.cool/arxiv/{arxiv_id})]\n\n"
    paper_string += f'**Authors:** {", ".join(authors)}\n\n'
    if "TLDR" in paper_entry:
        tldr = paper_entry["TLDR"]
        paper_string += f'**TLDR:** {tldr}\n\n'
    paper_string += f"<details>\n<summary><strong>Abstract</strong></summary>\n\n{abstract}\n\n</details>\n\n"
    if "COMMENT" in paper_entry:
        comment = paper_entry["COMMENT"]
        paper_string += f"**Comment:** {comment}\n\n"
    if "RELEVANCE" in paper_entry and "NOVELTY" in paper_entry:
        # get the relevance and novelty scores
        relevance = paper_entry["RELEVANCE"]
        novelty = paper_entry["NOVELTY"]
        paper_string += f"**Relevance:** {relevance}\n"
        paper_string += f"**Novelty:** {novelty}\n"
    topic_id = idx // topic_shift
    topic_str = f'topic-{topic_id}' if topic_id else 'go-beyond'
    paper_string += f"Back to [[topic](#{link_prefix}{topic_str})] [[top](#{link_prefix}topics)]\n"
    return paper_string


def render_title_and_author(paper_entry: dict, idx: int) -> str:
    # get the arxiv id
    arxiv_id = paper_entry["arxiv_id"]
    # get the title
    title = paper_entry["title"]
    # get the arxiv url
    arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
    authors = paper_entry["authors"]

    raw_title_url = f'{idx} {title}'
    # Keep only English letters, numbers, and spaces
    cleaned = re.sub(r'[^a-zA-Z0-9 -]', '', raw_title_url)

    # Replace spaces with dashes
    cleaned = cleaned.replace(' ', '-').lower()
    paper_string = f'{idx}\. [{title}]({arxiv_url}) [[more](#{link_prefix}{cleaned})] \\\n'
    paper_string += f'**Authors:** {", ".join(authors)}\n'
    return paper_string


def render_criteria(criteria: list[str]) -> str:
    criteria_string = ""
    for criterion in criteria:
        topic_idx = int(criterion.split('.')[0])
        criteria_string += f"[{criterion}](#{link_prefix}topic-{topic_idx})\n\n"
    criteria_string += f'[Go beyond](#{link_prefix}go-beyond)\n\n'
    return criteria_string

def extract_criterion_from_paper(paper_entry: dict) -> list:
    if "COMMENT" not in paper_entry:
        return [0]
    comment = paper_entry["COMMENT"]
    # Match 'Criterion' or 'criterion', ASCII or full-width colon, numbers separated by ASCII or Chinese commas
    match = re.search(
        r'criterion\s*[:：]\s*([0-9]+(?:\s*[ ,，]\s*[0-9]+)*)',
        comment,
        re.IGNORECASE
    )
    if not match:
        return [0]
    # Split on both ASCII comma and Chinese comma, strip spaces
    parts = re.split(r'[ ,，]+', match.group(1))
    numbers = [int(p) for p in parts if p.isdigit()]
    return numbers if numbers else [0]

def render_md_paper_title_by_topic(topic, paper_in_topic: list[str], filtered_criteria=None) -> str:
    topic_title = ""
    if filtered_criteria and topic.startswith("Topic "):
        # Extract the topic index from "Topic {i}"
        topic_idx = int(topic.split(" ")[1])
        if 1 <= topic_idx <= len(filtered_criteria):
            # Extract the topic title from filtered_criteria
            topic_title = filtered_criteria[topic_idx-1].strip()

    paper_count = len(paper_in_topic)

    # Add explicit anchor ID for topic sections
    if topic.startswith("Topic "):
        topic_idx = int(topic.split(" ")[1])
        anchor_id = f'<a id="topic-{topic_idx}"></a>\n'
    elif topic == "Go beyond":
        anchor_id = f'<a id="go-beyond"></a>\n'
    else:
        anchor_id = ""

    if topic_title:
        return anchor_id + f"### {topic}: {topic_title} ({paper_count} papers)\n" + "\n".join(paper_in_topic) + f"\n\nBack to [[top](#{link_prefix}topics)]\n\n---\n"
    else:
        return anchor_id + f"### {topic} ({paper_count} papers)\n" + "\n".join(paper_in_topic) + f"\n\nBack to [[top](#{link_prefix}topics)]\n\n---\n"


def render_md_string(papers_dict):
    # header
    with open("configs/paper_topics.txt", "r") as f:
        criteria = f.readlines()

    filtered_criteria = [i for i in criteria if len(i.strip()) and i.strip()[0] in '0123456789']

    criteria_string = render_criteria(filtered_criteria)

    import random
    def generate_background_for_white_foreground(threshold:int=150):
        # Ensure that the color is dark enough for white text to be readable
        # by keeping the RGB values below a certain threshold (e.g., 200)
        r = random.randint(0, threshold)
        g = random.randint(0, threshold)
        b = random.randint(0, threshold)

        # Convert the RGB values to a hexadecimal string
        hex_color = f'{r:02x}{g:02x}{b:02x}'
        return hex_color

    random_font = random.sample(['Cookie', 'Lato', 'Arial', 'Comic', 'Inter', 'Bree', 'Poppins'], 1)[0]
    random_emoji = random.sample("😸😺🐱🐶🐼🐰🐥🐢🐣🌸🍀🌈☀️🍓🍦🍪🐻🦊🦄🐣🐤🐦🐧🦉🐸🐞🦋🐝🍄🌻🌷🌼🌺🌿🍃🍒🍑🍍🍉🍌🍫🍬🍭🍯🍼🧸🎀🎈🎉🛀🎁💐💖💕💞💓💗💘💝💟💌🎊🍩🍨🍧🍡🍖🍗🍕🍔🍟🌭🍿🍱🍣🍤🍥🍚🍙🍘🍜🍝🍛🍢🍵🍶🥂🥤🍹🍺🍻🥃🍷🥄🍴🍽🥢🥡🥪🥗🥘🥓🥞🥐🥖🥨🥯🥚🥦🥒🥑🥔🥕🥗🥝🥭🥥🥬🥪🥫🥟🥠🥡🥢🥣🥤🥧🥨🥩🥪🥫🥬🥭🥮", 1)[0]

    output_string = (
        "# Personalized Daily Arxiv Papers "
        + datetime.today().strftime("%m/%d/%Y")
        + "\n\nThis project is adapted from [tatsu-lab/gpt_paper_assistant](https://github.com/tatsu-lab/gpt_paper_assistant). The source code of this project is at [lxysl/gpt_paper_assistant](https://github.com/lxysl/gpt_paper_assistant)\n\n"
        + "\n\n## Topics\n\nPaper selection prompt and criteria (jump to the section by clicking the link):\n\n"
        + criteria_string
        + "\n---\n"
    )

    # Initialize the list of papers for each topic (including the "Go beyond" category)
    paper_full_group_by_topic = [[] for _ in range(len(filtered_criteria) + 1)]
    
    # Track processed papers to avoid duplicates in the "Go beyond" category
    processed_beyond_papers = set()
    
    # Process each paper
    for i, (paper_id, paper) in enumerate(papers_dict.items()):
        topic_indices = extract_criterion_from_paper(paper)
        
        for topic_idx in topic_indices:
            if topic_idx > len(filtered_criteria):
                topic_idx = 0
            
            # If the paper is classified into a specific topic, or the "Go beyond" category but not yet processed
            if topic_idx > 0 or (topic_idx == 0 and paper_id not in processed_beyond_papers):
                idx = i + topic_idx * topic_shift
                full_string = render_paper(paper, idx)
                paper_full_group_by_topic[topic_idx].append(full_string)
                
                # If the paper is classified into a specific topic, mark it as processed (should not appear in the "Go beyond" category)
                if topic_idx > 0:
                    processed_beyond_papers.add(paper_id)

    # Render today's Spotlight papers (high RELEVANCE AND high NOVELTY), with jump links to each paper
    key_papers_string = ""
    for i, (paper_id, paper) in enumerate(papers_dict.items()):
        # Only highlight papers with both high relevance (≥9) and high novelty (≥8), or perfect relevance (10)
        if (paper["RELEVANCE"] >= 9 and paper["NOVELTY"] >= 8) or paper["RELEVANCE"] == 10:
            topic_indices = extract_criterion_from_paper(paper)
            for topic_idx in topic_indices:
                idx = i + topic_idx * topic_shift
                key_papers_string += f'{paper["title"]} [topic {topic_idx}] [[jump](#{link_prefix}paper-{idx})]\\\n'
    output_string += f"## Today's Spotlight Papers\n\n{key_papers_string}\n\n---\n\n"

    # Render each topic's content
    for topic_idx, paper_in_topic in enumerate(paper_full_group_by_topic):
        if topic_idx == 0:
            # Skip the "Go beyond" category, handle it later.
            continue
        output_string += render_md_paper_title_by_topic(f'Topic {topic_idx}', paper_in_topic, filtered_criteria)
    # Render the "Go beyond" topic last
    output_string += render_md_paper_title_by_topic("Go beyond", paper_full_group_by_topic[0], filtered_criteria)

    return output_string


if __name__ == "__main__":
    # parse output.json into a dict
    with open("out/output.json", "r") as f:
        output = json.load(f)
    # write to output.md
    with open("out/output.md", "w") as f:
        f.write(render_md_string(output))
