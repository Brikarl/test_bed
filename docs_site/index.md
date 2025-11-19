---
layout: default
title: Home
nav_order: 1
description: "Enterprise-grade AI system engineering testbed - from classical search to cutting-edge LLM training"
permalink: /
---

# AI Engineering Testbed
{: .fs-9 }

A comprehensive platform for exploring enterprise-grade AI system engineering, demonstrating end-to-end MLOps workflows from classical search systems to cutting-edge LLM training and intelligent automation.
{: .fs-6 .fw-300 }

[Get Started](#-quick-start){: .btn .btn-primary .fs-5 .mb-4 .mb-md-0 .mr-2 }
[View on GitHub](https://github.com/tylerelyt/test_bed){: .btn .fs-5 .mb-4 .mb-md-0 }

---

## Why "Testbed"?
{: .text-delta }

The name **Testbed** carries a rich heritage from engineering disciplines:

- **Electrical Engineering Era**: Physical test benches for circuit validation and prototyping
- **Computer Science Evolution**: Software testing frameworks and validation environments  
- **AI Engineering Today**: Integrated platform for experimenting with end-to-end AI systems

This project embodies the engineering philosophy of **learning through hands-on experimentation** - providing a controlled environment to explore, validate, and understand complex AI architectures before production deployment.

{: .note }
> A testbed isn't just a testing tool - it's a **learning platform** where theory meets practice, and where mistakes become valuable insights.

### Industry Practice: Internal Portals at Scale

This architecture mirrors how leading AI companies operate. Before any new AI feature reaches production, it goes through internal validation platforms:

- 🔬 **Internal Portal**: Dashboard for research teams to experiment and iterate
- 🔍 **X-Ray Interface**: Deep debugging and observability tools for developers  
- 📊 **A/B Testing Hub**: Controlled experiments before public rollout
- 🎯 **Staging Environment**: Production-like testing without user impact

**Our Testbed Dashboard = Industry Internal Research Portal**

Major tech giants - both domestic and international - all follow this pattern. This project brings those enterprise-grade internal tooling practices to the open-source community, demonstrating how production AI systems are validated and refined before public deployment.

---

## Platform Overview
{: .text-delta }

**Full-Stack AI System** covering five major domains:

🔍 **Search & Recommendation**
{: .label .label-blue }
CTR prediction, A/B testing, knowledge graphs, and intelligent ranking
{: .fs-3 }

🤖 **LLMOps Training Pipeline**
{: .label .label-green }
Complete CPT → SFT → DPO workflow with online feedback loops
{: .fs-3 }

💬 **Context Engineering**
{: .label .label-purple }
MCP-based context orchestration with RAG capabilities for intelligent Q&A
{: .fs-3 }

🖼️ **Multimodal AI**
{: .label .label-yellow }
CLIP-based image search and cross-modal understanding
{: .fs-3 }

🖱️ **GUI Automation Agent**
{: .label .label-red }
OSWorld-based desktop task automation with VLM reasoning
{: .fs-3 }

---

## 🚀 Quick Start
{: .text-delta }

### Installation

```bash
# Clone the repository
git clone https://github.com/tylerelyt/test_bed.git
cd test_bed

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Launch the System

```bash
# Method 1: Using launch script
./quick_start.sh

# Method 2: Direct launch
python start_system.py
```

{: .note }
> After the system starts, visit [http://localhost:7861](http://localhost:7861) to access the interface.

---

## 📚 Documentation Navigation
{: .text-delta }

<div class="code-example" markdown="1">

### Core Modules

[Search & Recommendation System]({{ site.baseurl }}/docs/search-recommendation){: .btn .btn-outline }
- [CTR Prediction Models]({{ site.baseurl }}/docs/search-recommendation/ctr-prediction)
- [Model Evaluation]({{ site.baseurl }}/docs/search-recommendation/model-evaluation)
- [Interpretability Analysis]({{ site.baseurl }}/docs/search-recommendation/interpretability)
- [Fairness Analysis]({{ site.baseurl }}/docs/search-recommendation/fairness)
- [AutoML Optimization]({{ site.baseurl }}/docs/search-recommendation/automl)

[LLMOps Training Pipeline]({{ site.baseurl }}/docs/llmops){: .btn .btn-outline }

[Context Engineering]({{ site.baseurl }}/docs/rag){: .btn .btn-outline }

[Multimodal AI]({{ site.baseurl }}/docs/multimodal){: .btn .btn-outline }
- [Image Search]({{ site.baseurl }}/docs/multimodal/image-search)
- [Image Generation]({{ site.baseurl }}/docs/multimodal/image-generation)

[GUI Automation Agent]({{ site.baseurl }}/docs/gui-agent){: .btn .btn-outline }

[Model Serving]({{ site.baseurl }}/docs/model-serving){: .btn .btn-outline }

</div>

---

## 🛠️ Technology Stack
{: .text-delta }

| Category | Technologies |
|:---------|:------------|
| **Classical ML** | scikit-learn (Logistic Regression), TensorFlow (Wide & Deep) |
| **Large Language Models** | LLaMA-Factory, LoRA, Ollama, OpenAI API |
| **Training Techniques** | CPT, SFT, DPO, RLHF-free alignment |
| **Computer Vision** | OpenAI CLIP (ViT-B/32), Hugging Face Transformers |
| **Vision-Language Models** | Qwen-VL, GPT-4V, QVQ |
| **Web Framework** | Gradio (responsive UI), Flask (REST API) |
| **Model Serving** | Independent process, RESTful endpoints |

---

## 🌟 Key Features

- ✅ **Production-Ready Architecture**: Microservice design with independent model serving
- ✅ **Complete Observability**: Full monitoring and logging infrastructure
- ✅ **Automated Experiment Management**: A/B testing framework and result tracking
- ✅ **Real-World Deployment Patterns**: Scalability considerations and best practices
- ✅ **Educational Resources**: Comprehensive documentation and examples

---

## 🌐 Why English Matters
{: .text-delta }

### The Source of Latest Information

**English is the primary language of cutting-edge AI research and development**. Most of the latest technical breakthroughs, research papers, engineering blogs, and best practices are published in English first.

**Why This Matters for AI Professionals**:

1. **Latest Research**: 
   - New papers on arXiv are primarily in English
   - Breakthrough announcements from leading labs (OpenAI, Anthropic, Google) are in English
   - Technical discussions and insights happen in English first

2. **Engineering Best Practices**:
   - Industry engineering blogs (Anthropic, OpenAI, Google AI) share production insights in English
   - Open-source documentation and discussions are predominantly in English
   - Conference talks and tutorials are primarily in English

3. **Timeliness**:
   - English sources provide immediate access to new information
   - Translations often lag behind, missing critical updates
   - Direct access avoids potential misunderstandings from translation

4. **Career Growth**:
   - International collaboration requires English proficiency
   - English skills enable participation in global AI community

### Learning Resources

We've curated essential English learning resources that combine **language improvement** with **cutting-edge technical knowledge**:

👉 **[Learning Resources →]({{ site.baseurl }}/docs/learning-resources)** - Essential engineering blogs and learning strategies

**Featured Resources**:
- **Anthropic Engineering Blog**: Deep dives into AI safety and scaling
- **OpenAI Developer Blog**: API updates and best practices  
- **Google AI Technology Blog**: Latest research breakthroughs

These resources provide the **dual benefit** of improving your English while staying current with the latest AI developments.

{: .note }
> **Tip**: Reading technical blogs in English is one of the most effective ways to improve both your language skills and technical knowledge simultaneously. Start with summaries, focus on technical terms, and practice regularly.

---

## 📄 License

This project is distributed under the MIT License - see the [LICENSE](https://github.com/tylerelyt/test_bed/blob/main/LICENSE) file for details.

## 🤝 Contributing

Issues and Pull Requests are welcome! Please check our [Contributing Guidelines](https://github.com/tylerelyt/test_bed/blob/main/CONTRIBUTING.md).

## 📞 Contact

- **Project Homepage**: [https://github.com/tylerelyt/test_bed](https://github.com/tylerelyt/test_bed)
- **Issue Tracker**: [https://github.com/tylerelyt/test_bed/issues](https://github.com/tylerelyt/test_bed/issues)

