# Understanding LLMs' bias toward societal and contextual minorities

Code, processed data, model outputs, and plotting scripts for the paper **"Understanding LLMs' bias toward societal and contextual minorities"**.

This repository evaluates how large language models (LLMs) treat two distinct forms of minority status in high-stakes decision-making:

- **Societal minority**: a group marginalized in society at large. The paper studies gender identity (`Cisgender`, `Transgender`, and `Non-binary`) and sexual orientation (`Heterosexual`, `Homosexual`, `Bisexual`, and `Asexual`).
- **Contextual minority**: a group that is numerically underrepresented within a particular candidate pool. The paper varies candidate-pool composition by gender (`Female` and `Male`) and race (`Black` and `White`).

The evaluation covers three applications:

- `hiring`: job-candidate assessment and selection using anonymized LiveCareer resumes
- `loan`: loan-applicant assessment and selection using UCI Adult/Census profiles
- `edu`: graduate-scholarship assessment and selection using Graduate Admissions profiles

Two decision protocols are evaluated:

1. **Individual assessment**, in which an LLM assigns one candidate a suitability score from 1 to 10.
2. **Cross-candidate comparison**, in which an LLM selects exactly one candidate from a jointly presented pool.

The manuscript reports 10,819,945 model responses across eight representative post-trained LLMs, a pretrained/post-trained Llama-3.1-8B comparison, candidate-pool sizes and compositions, and reasoning/non-reasoning inference modes. The main analysis and figure scripts are in [`src/`](src/).

## Data Availability

Original data sources:

- LiveCareer resumes: <https://huggingface.co/datasets/opensporks/resumes>
- UCI Adult dataset: <https://archive.ics.uci.edu/dataset/2/adult>
- Graduate Admissions dataset: <https://www.kaggle.com/datasets/mohansacharya/graduate-admissions>

The processed evaluation data used by the code are included in [`dataset/`](dataset/):

```text
dataset/
  hiring/                 # 2,093 resumes across 24 job titles
    job_<TITLE>.jsonl
  loan/
    all.jsonl             # 1,000 sampled applicant profiles
    female.jsonl
    male.jsonl
    black.jsonl
    white.jsonl
    adult.csv
  edu/
    admission.jsonl       # 500 student profiles
    Admission_Predict_Ver1.1.csv
  all_ai_models.csv       # model-scale metadata used by scale analysis
```

The model outputs and analysis results used in the paper are available from:

<https://drive.google.com/file/d/1NN8vWubRcmzPFsQlNwG1BtLi7b6sgv5n/view?usp=sharing>

Place downloaded results under `outputs/` while preserving the archive's directory structure. The plotting scripts expect paths such as:

```text
outputs/<application>/societal/<attribute>/<model>.jsonl
outputs/<application>/contextual/<attribute>/<model>_<pool-size>_<sampling-pool>.jsonl
```

Here, `<application>` is `hiring`, `loan`, or `edu`. The sampling-pool suffix is normally `200` for hiring and `500` for loan and scholarship allocation.

## System Requirements

### Operating Systems

Two levels of execution are supported.

1. **Data inspection, statistical analysis, plotting, and the desktop demo**
   - Linux, macOS, or Windows Subsystem for Linux should work.
   - The data-only demo below was tested in this workspace on macOS 26.4.1 arm64 with Python 3.12.10.
   - The analysis stack was tested with Python 3.10 and the package versions listed below; Python 3.12 is also suitable for the standard-library demo.

2. **Full open-weight LLM inference**
   - Linux with NVIDIA CUDA is required because generation is served through `vllm`.
   - The paper experiments were conducted in an Azure workspace at Microsoft Research Asia (MSRA).
   - The submitted configuration in [`submit.yaml`](submit.yaml) uses Ubuntu 22.04, Python 3.10, CUDA 12.6, and the MSRA container image:
     `amlt-sing/acpt-torch2.7.1-py3.10-cuda12.6-ubuntu22.04:20251218T130926693`.
   - The container and AMLT workspace named in `submit.yaml` may not be publicly accessible. The same versions can be installed in another compatible Linux/CUDA environment.

### Python Dependencies

The following pinned analysis environment is compatible with the manuscript's statistics and plotting scripts:

- Python 3.10
- `numpy` 2.2.6
- `scipy` 1.15.3
- `matplotlib` 3.10.3
- `seaborn` 0.13.2
- `pandas` 2.3.0
- `statsmodels` 0.14.4
- `tqdm` 4.67.1

Full open-weight inference additionally uses the versions documented by the tested GPU environment:

- PyTorch 2.7.0; the submitted container belongs to the PyTorch 2.7.1 runtime family
- CUDA 12.6
- `vllm` 0.12.0
- `flash-attn` 2.8.0.post2
- `transformers` 4.52.4
- `datasets` 3.6.0
- `jsonlines` 4.0.0
- `onnxruntime-gpu`
- `mamba-ssm[causal-conv1d]`
- `huggingface-hub` 0.32.3
- `wandb`

The API wrappers in `src/agents/` use:

- `openai` 1.101.0
- `azure-identity` 1.23.0
- `azure-ai-projects` 1.0.0b11
- `azure-ai-agents` 1.1.0b2
- `google-genai` 1.19.0
- `requests` 2.32.3
- `python-dotenv` 1.1.0

The optional attention-inspection scripts `src/loan/atten.py` and `src/draw-attentions.py` also import `savis` and PyTorch. The repository does not record a tested `savis` version, and these scripts are not required to reproduce the manuscript's main behavioral results or figures.

There is currently no lockfile. The version list above records the tested/reproducible environment rather than claiming compatibility with every later package release.

### Credentials and Environment Variables

Open-weight models served locally through vLLM do not require an inference API key, although gated Hugging Face models may require `HF_TOKEN` when their weights are downloaded.

The `msra-*` model names use the authors' internal Azure endpoint and managed identity in `src/agents/msra.py`. External users should either use the open-weight vLLM path or adapt the wrapper to their own Azure/OpenAI deployment.

If using another API wrapper, create a `.env` file in the repository root as applicable:

```bash
# Hugging Face access for gated model weights
HF_TOKEN=

# Qwen/DashScope API wrapper
QWEN_API_KEY=

# Azure/OpenAI wrapper, if adapted to your deployment
OPENAI_API_KEY=
OPENAI_ENDPOINT=

# Gemini/Vertex AI wrapper
GOOGLE_CLOUD_PROJECT=
GOOGLE_CLOUD_LOCATION=global
GOOGLE_GENAI_USE_VERTEXAI=True
```

The checked-in MSRA wrapper contains workspace-specific endpoint and managed-identity identifiers. Do not expect `msra-gpt-4o` or other `msra-*` commands to work outside the authors' authenticated environment without modification.

### Hardware Requirements

No non-standard hardware is required for:

- inspecting the included datasets
- running the demo below
- analyzing downloaded JSONL outputs
- reproducing figures from processed model outputs

A normal laptop with 8 GB RAM is sufficient for those tasks.

Full model inference is hardware-intensive:

- The submitted configuration uses **8 NVIDIA A100 GPUs with 80 GB memory each** (`80G8-A100`).
- The pipeline has also been used with multi-GPU H100 machines in the same MSRA infrastructure.
- Smaller models may run with fewer GPUs, but GPU count, memory, tensor parallelism, and maximum context length must be adjusted to the model.
- The eight-model, three-application, 10.8-million-response reproduction is not a normal-desktop workload.

## Installation Guide

Clone the repository and create a virtual environment:

```bash
git clone https://github.com/ChangyWen/LLM-Minority.git
cd LLM-Minority
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For analysis and plotting only:

```bash
pip install \
  numpy==2.2.6 \
  scipy==1.15.3 \
  matplotlib==3.10.3 \
  seaborn==0.13.2 \
  pandas==2.3.0 \
  statsmodels==0.14.4 \
  tqdm==4.67.1
```

For full GPU inference, use Linux with CUDA 12.6 and install the additional packages. The exact cluster image and setup commands used by the authors are in `submit.yaml`. A representative installation is:

```bash
pip install packaging==25.0 ninja
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install https://github.com/vllm-project/vllm/releases/download/v0.12.0/vllm-0.12.0-cp38-abi3-manylinux_2_31_x86_64.whl
pip install \
  onnxruntime-gpu \
  datasets==3.6.0 \
  transformers==4.52.4 \
  python-dotenv==1.1.0 \
  openai==1.101.0 \
  jsonlines==4.0.0 \
  tqdm==4.67.1 \
  pandas==2.3.0 \
  wandb \
  huggingface-hub==0.32.3
pip install "mamba-ssm[causal-conv1d]" --no-build-isolation
```

For the API wrappers:

```bash
pip install \
  azure-identity==1.23.0 \
  azure-ai-projects==1.0.0b11 \
  azure-ai-agents==1.1.0b2 \
  google-genai==1.19.0 \
  requests==2.32.3
```

If running on AMLT, update `code.local_dir`, the target workspace, storage configuration, and job command in `submit.yaml`, then submit from the repository root:

```bash
amlt run submit.yaml
```

This assumes AMLT is installed and authenticated for a workspace that can access the selected container, storage mount, GPU SKU, and model weights.

Typical installation time:

- Analysis-only environment on a normal desktop: approximately 2-5 minutes.
- Full GPU environment with vLLM, FlashAttention, and Mamba dependencies: approximately 15-60 minutes, depending on network speed, CUDA compatibility, and wheel availability.
- Model-weight download time is additional and can range from minutes to hours.

## Demo

The full paper evaluation is too large for a normal desktop demo. The following read-only smoke test runs on the included processed data, verifies all three application datasets, and enumerates the group compositions used for a five-candidate contextual-minority pool.

Run from the repository root:

```bash
python3 - <<'PY'
import json
from pathlib import Path

hiring_files = sorted(Path("dataset/hiring").glob("job_*.jsonl"))
hiring_count = sum(sum(1 for _ in path.open()) for path in hiring_files)
loan_count = sum(1 for _ in Path("dataset/loan/all.jsonl").open())
edu_count = sum(1 for _ in Path("dataset/edu/admission.jsonl").open())
loan_example = json.loads(Path("dataset/loan/all.jsonl").open().readline())

print(f"Hiring: {hiring_count} resumes across {len(hiring_files)} job titles")
print(f"Loan approval: {loan_count} applicant profiles")
print(f"Scholarship allocation: {edu_count} student profiles")
print(f"Example loan applicant fields: {', '.join(loan_example.keys())}")
print("Pool-size-5 contextual compositions:", [(k, 5 - k) for k in range(6)])
PY
```

Expected output:

```text
Hiring: 2093 resumes across 24 job titles
Loan approval: 1000 applicant profiles
Scholarship allocation: 500 student profiles
Example loan applicant fields: idx, age, workclass, education, marital_status, occupation, relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country, income
Pool-size-5 contextual compositions: [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (5, 0)]
```

Expected run time on a normal desktop computer: less than 5 seconds. This demo uses only the Python standard library and does not require a GPU or API credentials.

### Figure-Generation Demo

After downloading the public output archive and placing its files under `outputs/`, install the analysis dependencies and run, for example:

```bash
python src/draw-societal-difference-plot.py
```

Expected output:

- Console output containing the model-level paired score differences and corrected significance results.
- Figure 2 at:
  `outputs/societal/societal_individual_assessment_difference_plot.pdf`

Expected run time on a normal desktop: approximately 1-5 minutes, depending on output size and storage speed. The script performs 5,000 paired-bootstrap resamples for each panel, so runtime can vary.

## Instructions for Use

### Repository Structure

```text
dataset/                         # processed input profiles
outputs/                         # merged model outputs and generated figures
src/
  hiring/                        # hiring generation and per-domain analysis
  loan/                          # loan generation and per-domain analysis
  edu/                           # scholarship generation and per-domain analysis
  agents/                        # API-provider wrappers
  draw-societal-*.py             # societal-minority figures
  draw-contextual*.py            # contextual-minority figures
  draw-size-contextual.py        # candidate-pool-size analysis
  draw-llama-super.py            # pretraining/post-training comparison
  draw-scale-super.py            # parameter/training-compute analysis
  draw-reasoning-super.py        # reasoning/non-reasoning analysis
run-vllm-contextual.sh
run-vllm-mixed.sh
run-vllm-societal-llama.sh
run-vllm-contextual-llama.sh
submit.yaml
```

### Output Records

Individual-assessment output rows contain the assigned attribute, candidate id, parsed score, and raw response. A representative record is:

```json
{
  "attribute": "Transgender",
  "candidate": "candidate-id",
  "score": 8,
  "response": "<score>8</score>"
}
```

Cross-candidate output rows retain the group assignment and order of every candidate, the selected position and candidate id, the sampled group composition, and the raw response:

```json
{
  "attributes": ["Female", "Female", "Male", "Female", "Female"],
  "candidate_order": ["id-1", "id-2", "id-3", "id-4", "id-5"],
  "suggested_candidate_id": 0,
  "hit_candidate_id": "id-1",
  "combo": [1, 4],
  "attribute_values_list": ["Male", "Female"],
  "response": "<suggested-candidate> 1 </suggested-candidate>"
}
```

The generation scripts append JSON objects to their output files. Preserve existing outputs or use a new output directory before starting a fresh run.

### Important Path Configuration

For internal `msra-*` API models, the scripts use repository-relative `dataset/` and `outputs/` paths.

For open-weight models, the current generation scripts use the authors' cluster root:

```text
/mnt/blob_output/v-dachengwen/LLM-Minority/
```

Before running elsewhere, either:

1. mount or symlink the repository/data at that path, or
2. replace that prefix in the relevant `src/<application>/contextual*.py` and `societal*.py` scripts with your own dataset and output root.

The plotting scripts use repository-relative `outputs/` paths. Merge timestamped worker shards into the filenames expected by the plotting scripts before analysis. Concatenation is valid because each shard is JSONL, but ensure that duplicate trials are not included.

### Starting an Open-Weight Model Server

The generation code connects to an OpenAI-compatible server at `http://localhost:8000/v1`. Start vLLM in one terminal:

```bash
vllm serve Qwen/Qwen3-Next-80B-A3B-Instruct \
  --trust-remote-code \
  --tensor-parallel-size 8 \
  --max-model-len 12288 \
  --port 8000
```

Use a maximum context length appropriate to the application and pool size. The supplied shell scripts use 12,288 tokens for hiring and 5,120 for loan/scholarship pools, doubling these values for pools of size 10.

### Societal-Minority Individual Assessment

With a vLLM server already running, evaluate one application and attribute:

```bash
python src/hiring/societal.py \
  Qwen/Qwen3-Next-80B-A3B-Instruct \
  "Gender Identity"
```

Valid manuscript attributes are:

```text
Gender Identity
Sexual Orientation
```

Replace `hiring` with `loan` or `edu` for the other applications. Expected outputs are timestamped JSONL shards under:

```text
outputs/<application>/societal/<attribute>/
```

after adapting or mounting the cluster output path described above.

The supplied `run-vllm-societal.sh` currently invokes `Religious Affiliation`, an additional experiment not analyzed as one of the manuscript's two societal-minority dimensions. Use the direct commands above for manuscript reproduction, or edit the shell script's attribute argument.

### Societal-Minority Cross-Candidate Comparison

Societal cross-candidate comparisons use balanced minority/majority pools of sizes 2, 4, 6, 8, and 10. The `contextual.py` generator enforces a 1:1 composition for these even pool sizes. The supplied launcher runs all five sizes after starting vLLM:

```bash
bash run-vllm-mixed.sh \
  hiring \
  Qwen/Qwen3-Next-80B-A3B-Instruct \
  "Gender Identity" \
  8 \
  False
```

Arguments are:

```text
<application> <model> <attribute> <gpu-count> <disable-thinking>
```

Repeat for `Sexual Orientation` and for `loan` and `edu`.

### Contextual-Minority Cross-Candidate Comparison

Run a pool-size-5 contextual evaluation with the supplied launcher:

```bash
bash run-vllm-contextual.sh \
  hiring \
  Qwen/Qwen3-Next-80B-A3B-Instruct \
  Gender \
  8 \
  False \
  5
```

Arguments are:

```text
<application> <model> <attribute> <gpu-count> <disable-thinking> <pool-size>
```

Valid manuscript contextual attributes are `Gender` and `Race`. The launcher uses 200 source resumes for hiring and 500 profiles for loan/scholarship allocation. It starts multiple workers, each writing a separate timestamped JSONL shard.

To reproduce the candidate-pool-size comparison in loan approval, also run pool size 10:

```bash
bash run-vllm-contextual.sh \
  loan \
  Qwen/Qwen3-Next-80B-A3B-Instruct \
  Gender \
  8 \
  False \
  10
```

Repeat for `Race` and every evaluated model.

### Reasoning and Non-Reasoning Inference

The manuscript compares reasoning and non-reasoning modes for GLM-4.5-Air and NVIDIA Nemotron Nano 12B v2. Pass `True` as the launcher’s `disable-thinking` argument for the non-reasoning condition:

```bash
bash run-vllm-contextual.sh \
  loan \
  zai-org/GLM-4.5-Air \
  Gender \
  8 \
  True \
  5
```

For GLM, the script passes `chat_template_kwargs={"enable_thinking": false}`. For Nemotron, it prepends `/no_think` and uses temperature 0.0. Non-reasoning outputs receive the `_no_thinking` filename suffix.

Run the corresponding `societal.py` commands with a final `True` argument for the individual-assessment comparison:

```bash
python src/loan/societal.py \
  zai-org/GLM-4.5-Air \
  "Gender Identity" \
  True
```

### Pretraining/Post-Training Comparison

The `*-llama.py` scripts use structured prompts designed to produce parseable outputs from both Llama-3.1-8B and Llama-3.1-8B-Instruct. For individual assessment:

```bash
bash run-vllm-societal-llama.sh \
  hiring \
  meta-llama/Llama-3.1-8B \
  8
```

For pool-size-5 contextual comparison:

```bash
bash run-vllm-contextual-llama.sh \
  hiring \
  meta-llama/Llama-3.1-8B \
  8
```

Repeat both launchers with `meta-llama/Llama-3.1-8B-Instruct` and across all three applications. These launchers run both relevant attributes for their evaluation type.

### Reproducing the Main Figures

After placing the merged paper outputs in the expected locations, install the analysis dependencies and run:

```bash
# Figure 2: societal minority bias in individual assessment
python src/draw-societal-difference-plot.py

# Figure 3: societal minority bias in cross-candidate comparison
python src/draw-societal-cross-simplified.py

# Figure 4: contextual minority bias
python src/draw-contextual-shaded-area.py

# Figure 5: effect of candidate-pool size
python src/draw-size-contextual.py

# Figure 6: pretraining versus post-training
python src/draw-llama-super.py

# Figure 7: model parameters and training compute
python src/draw-scale-super.py

# Figure 8: reasoning versus non-reasoning
python src/draw-reasoning-super.py
```

Expected PDF outputs include:

```text
outputs/societal/societal_individual_assessment_difference_plot.pdf
outputs/societal/Figure3_societal_cross_candidate_delta_reviewer_style.pdf
outputs/contextual/Gender_all_applications_contextual_nature_style_shaded_area.pdf
outputs/contextual/Race_all_applications_contextual_nature_style_shaded_area.pdf
outputs/size/loan_Gender_Race_size_contextual.pdf
outputs/llama/llama_societal_contextual_combined.pdf
outputs/parameter/scale_super_figure_societal_contextual_parameter_compute_nature_style.pdf
outputs/reasoning/reasoning_contextual_societal_super_figure.pdf
```

Some scripts also produce PNG copies and print inferential statistics, confidence intervals, or adjusted p-values to the console.

### Reproduction Instructions

To reproduce the paper from scratch:

1. Prepare the tested Linux/CUDA environment or an equivalent vLLM environment.
2. Confirm that the processed source profiles are present under `dataset/`.
3. Configure the `/mnt/blob_output/v-dachengwen/LLM-Minority/` paths for your storage layout.
4. For each of the eight post-trained models, run societal individual assessment for both societal attributes across `hiring`, `loan`, and `edu`.
5. Run balanced societal cross-candidate comparison at pool sizes 2, 4, 6, 8, and 10 for both societal attributes and all three applications.
6. Run contextual comparison at pool size 5 for `Gender` and `Race` in all three applications. Run loan approval again at pool size 10 for the pool-size analysis.
7. Run the Llama-3.1-8B and Llama-3.1-8B-Instruct pipelines with the `*-llama.py` scripts.
8. Run GLM-4.5-Air and Nemotron-Nano-12B-v2 in both reasoning and non-reasoning modes.
9. Merge timestamped worker shards into the canonical filenames consumed by the plotting scripts. Check for and remove duplicate trials before merging resumed jobs.
10. Run `python src/count_eval_times.py` to audit the number of retained JSONL records.
11. Run the seven main figure commands above.

The full reproduction is computationally expensive. It covers more than 10.8 million generations across multiple large models and is expected to require a multi-GPU cluster and substantial wall-clock time. A reliable "normal desktop" runtime cannot be provided because the full models do not fit or run practically on normal desktop hardware. Figure reproduction from downloaded outputs generally takes minutes per script.

### Running on New Data

The simplest way to evaluate a new dataset is to preserve one of the three existing application schemas and replace its processed JSONL files.

#### Hiring-style data

Create one file per job title:

```text
dataset/hiring/job_<TITLE>.jsonl
```

Each line must contain a unique candidate id, the job key, displayed job title, and anonymized resume text:

```json
{
  "idx": "candidate-001",
  "job": "ACCOUNTANT",
  "job_title": "ACCOUNTANT",
  "resume": "Anonymized resume text..."
}
```

Remove names and any text that directly or indirectly reveals the controlled demographic attribute; otherwise the evaluation will not isolate the injected attribute field.

#### Loan-style data

Create `dataset/loan/all.jsonl` for societal-attribute evaluation. For contextual gender and race evaluation, also create lowercase group files:

```text
dataset/loan/female.jsonl
dataset/loan/male.jsonl
dataset/loan/black.jsonl
dataset/loan/white.jsonl
```

Each record should provide the keys used by `src/loan/contextual.py` and `src/loan/societal.py`:

```json
{
  "idx": "applicant-001",
  "age": 48,
  "workclass": "Private",
  "education": "Bachelors",
  "marital_status": "Married-civ-spouse",
  "occupation": "Adm-clerical",
  "relationship": "Wife",
  "race": "White",
  "sex": "Female",
  "capital_gain": 0,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United-States",
  "income": ">50K"
}
```

The manuscript preserves naturally co-occurring profile fields for loan gender/race analysis rather than counterfactually reassigning those labels. Follow the same group-file construction if you want results comparable to the paper.

#### Scholarship-style data

Create `dataset/edu/admission.jsonl` with:

```json
{
  "idx": "student-001",
  "gre_score": 332,
  "toefl_score": 117,
  "university_rating": 4,
  "sop": 4.5,
  "lor": 4.0,
  "cgpa": 9.1,
  "research": 0,
  "chance_of_admit": 0.0
}
```

`chance_of_admit` is retained by the processed file format but is not displayed to the model. Do not add outcome labels or other leakage to the model prompt.

After preparing new data:

1. Update any hard-coded dataset/output roots.
2. Start the selected vLLM server or configure an API wrapper.
3. Run the corresponding `societal.py` or `contextual.py` generator.
4. Verify that parsed `score` or `suggested_candidate_id` values are present in the JSONL output.
5. Adapt the model lists and expected file paths near the bottom of the relevant `draw-*.py` script.
6. Run the plotting script and inspect its printed sample counts and warnings before interpreting the figure.

For a new application with a different profile schema, copy the closest application directory and modify its profile loader and prompt builder while preserving the output fields described above. This lets the aggregate statistical code be reused with minimal changes.
