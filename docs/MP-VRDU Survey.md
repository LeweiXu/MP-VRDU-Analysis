## **Managing Evidence at Document Scale: A Survey of Multi-Page Visually Rich Document Understanding** 

**Lewei Xu**[1] **Yihao Ding**[1] _[∗]_ **Zihan Xu**[2] **Siwen Luo**[1] **Daochang Liu**[1] **Wei Liu**[1] 

1The University of Western Australia, Perth, Australia 

2The University of Melbourne, Melbourne, Australia 

> _∗Correspondence:_ yihao.ding@uwa.edu.au 

## **Abstract** 

Real-world documents often span tens to hundreds of pages, whereas most VRDU research still assumes a single-page setting. MP-VRDU is therefore a distinct problem rather than a simple extension: answer-relevant evidence is sparse, distributed across pages, and often exceeds a backbone MLLM’s context window. As a result, evidence management becomes the central design problem, as systems must decide which evidence to retain, discard, retrieve, or acquire during inference. We organise representative MP-VRDU systems along four axes: architecture, multimodal representation, inference-time adaptation, and training strategy. We further consolidate MP-VRDU datasets and finally, we identify open challenges in efficiency, faithfulness, evidence pipelines, supervision integrity, and reproducible evaluation that define the MP-VRDU frontier. 

## **1 Introduction** 

Visually-rich documents (VRDs) such as scientific papers, financial reports and technical manuals are pervasive in real-world workflows, and understanding them is a long-standing problem at the intersection of NLP and computer vision. Most prior work has focused on the single-page (SP) setting. However, real-world documents often span dozens of pages, with scientific papers referencing figures and tables many pages apart. Multi-page visually rich document understanding (MP-VRDU) is not a simple extension of SP-VRDU, since total document length frequently exceeds the backbone MLLM’s context window, evidence is sparsely distributed across pages, and document structure such as section hierarchies and cross-references cannot be recovered from per-token layout signals alone. 

MP-VRDU has seen rapid growth in recent years, with new architectures, training strategies, and inference methods emerging. Existing work has 

explored joint multi-page encoding (Tito et al., 2023), adaptation of pretrained MLLMs (Duan et al., 2025b), retrieval-augmented pipelines that decouple evidence selection from generation (Cho et al., 2024), and adaptive controllers that gather evidence iteratively at inference time (Sun et al., 2025), often combined with task-specific training or prompted reasoning loops (Xiong et al., 2026; Han et al., 2025). However, this body of work has developed along largely parallel lines, with limited cross-family comparison and inconsistent terminology, and few works explicitly distinguish what is genuinely MP-distinctive from what is inherited from the single-page setting. As a result, the field lacks a unified view of the design choices that matter most in multi-page settings, motivating the need for systematic organisation. 

Prior surveys have charted VRDU from complementary angles: OCR-centric deep learning pipelines (Subramani et al., 2020), multimodal pretraining (Ding et al., 2026), deep learning for information extraction over VRDs (Gbada et al., 2025), and recent LLM-era treatments (Ding et al., 2025) including retrieval-augmented generation for long documents (Gao et al., 2025). While valuable, these works remain anchored to the single-page setting, treating multi-page as an incremental extension rather than a distinct problem. 

This survey provides a systematic account of the emerging MP-VRDU frontier. First, we define MP-VRDU as a problem distinct from SPVRDU and organise existing systems through an **evidence-management perspective** , covering page-to-document encoding, MLLM-centric adaptation, retrieval-augmented and adaptive-trajectory approaches (§3). Second, we analyse how multipage systems represent and trade off text, visual, and structural modalities under document-length constraints (§4). Third, we review inference- and training-time strategies for scaling MP-VRDU, focusing on how systems retrieve, reason over, and 

1 

|**Model**|**Venue**|**OCR**|**Vision Encoder**|**LLM Backbone**|**Mod.**|**Search**|**Reasoning**|**Agentic**|
|---|---|---|---|---|---|---|---|---|
|**Page-to-Document Encoding Models**|||||||||
|Arctic-TILT (2025)|ACL|Yes|U-Net|T5-Large|T, V, L|-|-|-|
|GRAM (2024)|CVPR|Yes|DocFormerv2|DocFormerv2|T, V, L|-|-|-|
|Hi-VT5 (2023)|PR|Yes|DiT|T5-Base|T, V, L|-|-|-|
|RM-T5 (2024)|DAS|Yes|DiT|T5-Base|T, V, L|-|-|-|
|**MLLM-Centric Adaptation Models**|||||||||
|DocR1 (2026)|AAAI|No|Qwen2.5-VL|Qwen2.5-VL-7B|V|-|CoT, HD|-|
|DocSeeker (2025)|CVPR|No|Qwen2.5-VL ViT|Qwen2.5-VL-7B-Instruct|V|-|CoT, HD|-|
|Leopard (2024)|TMLR|No|SigLIP-SO400M|LLaMa-3.1-8B, Mistral-7B|V|-|CoT|-|
|mPLUG-DocOwl2 (2025)|ACL|No|ViT-L/14|mPLUG-Owl2|V|-|-|-|
|URaG (2026)|AAAI|No|Qwen2.5-VL ViT|Qwen2.5-VL-3B / Qwen2.5-VL-7B|V|-|HD|-|
|Docopilot (2025b)|CVPR|No*|InternViT-300M|Multiple (Intern)|V|-|-|-|
|Texthawk2 (2024)|preprint|No*|SigLIP-SO400M|Qwen2-7B-Instruct|V|-|-|-|
|DocSLM (2025)|CVPR|Yes|SigLIP2|Qwen2.5-1.5B|T, V, L|-|-|-|
|InstructDoc (2024)|AAAI|Yes|CLIP|FlanT5, BLIP2|T, V, L|-|-|-|
|LayTokenLLM (2025b)|CVPR|Yes|-|LLaMA3-8B, Qwen1.5-7B|T, L|-|-|-|
|CoR (2026)|preprint|Yes*|Qwen2.5-VL|Qwen2.5-VL-7B|T, V|-|CoT, PE, HD|-|
|**Retrieval-Augmented Pipelines**|||||||||
|AVIR (2025)|MMAsia|No|Pix2Struct|Qwen2.5-VL-3B-AWQ|V|D|-|-|
|DFVC (2026)|Electronics|No|VisRAG, ColQwen|Qwen2-VL-2B-Instruct|V|D|-|-|
|DREAM (2025)|MM|No|InternVL2 ViT|InternVL2-4B|V|J|HD|-|
|M3DocRAG (2024)|preprint|No|ColPali, ColQwen|Multiple (Qwen/Intern/Idefcs)|V|D|-|-|
|MM-R5 (2026)|AAAI|No|Qwen2.5-VL ViT|Qwen2.5-VL-7B|V|D|CoT|-|
|MoLoRAG (2025b)|EMNLP|No|ColPali|Multiple (Qwen/DS/LLaVA)|V|D, Nav|-|1A|
|Self-Attention Scoring (2024)|ICDAR|No|Pix2Struct|Pix2Struct|V|D|-|-|
|SLEUTH (2025)|preprint|No|ColPali, Qwen3-VL ViT|Multiple (Qwen/GLM/Gemini)|V|D|HD|4A|
|SV-RAG (2025)|ICLR|No|InternVL2, Phi-3-vision|Multiple (Intern/Pali/Phi)|V|D|-|-|
|VDocRAG (2025)|CVPR|No*|Phi3V-4.2B|Phi3V-4.2B|V|D|-|-|
|CREAM (2024)|MM|Yes|Pix2Struct|LLaMA2-7B|T, V|J|HD|-|
|KGP (2024c)|AAAI|Yes|-|Multiple|T, L|S, IR, Nav|-|1A|
|MHier-RAG (2025)|preprint|Yes|Qwen-VL-Plus|Multiple (Qwen/DS/GPT)|T, V, L|J|CoT, HD|-|
|MLDocRAG (2026)|preprint|Yes|Qwen2.5-VL|Multiple (Qwen)|T, V, L|D|CoT, HD|-|
|MultiDocFusion (2025)|ACL|Yes|-|Multiple (Qwen/LLaMa/Mistral)|T, V, L|D|HD|-|
|M2RAG (2025a)|NeurIPS|Yes*|VisRAG-Ret|Multiple (Qwen)|T, V|J|-|3A|
|MDocAgent (2025)|preprint|Yes*|ColPali, ColQwen2|Multiple (Llama/Qwen)|T, V|J|-|5A|
|PDF-WuKong (2024)|preprint|Yes*|IXC2-VL-4KHD|IXC2-VL-4KHD|T, V|J|-|-|
|RAG-DocVQA (2025)|ICDAR|Yes*|DiT, Pix2Struct|Qwen2.5-VL-7B-Instruct|T, V, L|D|-|-|
|VisDoMRAG (2025)|NAACL|Yes*|ColPali / ColQwen2|Multiple(Qwen/Gemini/GPT)|T, V|J|CoT|3A, SC|
|**Adaptive-Trajectory Pipelines**|||||||||
|Doc-V* (2026)|preprint|No|Qwen2.5-VL|Qwen2.5-VL-7B-Instruct|V|D, IR, QR, Nav|ReAct, HD|1A|
|DocReact (2025a)|ACL|No|ColPali, VisRAG|GPT-4o|V|D, IR, QR|ReAct, QD|1A|
|MACT (2025)|preprint|No|Multiple|Multiple (Qwen/MiMo/Intern)|V|D|CoT, PE|4A, SC, SV|
|VRAG-RL (2026)|NeurIPS|No|Qwen2.5-VL ViT|Qwen2.5-VL-3B/7B-Instruct|V|D, IR, QR|ReAct, HD|1A|
|DocLens (2025a)|preprint|Yes|–|Multiple (GPT/Gemini/Claude)|T, V, L|J|CoT|4A, SA|
|DMAP (2026)|WWW|Yes*|ColPali|GPT-4o / Qwen-plus|T, V, L|J|HD|2A, SR|
|DocAgent (2025)|EMNLP|Yes*|–|Multiple (GPT/Gemini/Claude)|T, V, L|J, IR, Nav|ReAct, HD|3A|
|DocDancer (2026)|preprint|Yes*|Qwen3-VL-235B|Qwen3-4B/30B-A3B|T, V, L|J, IR, QR|ReAct|1A|
|MM-Doc-R1 (2026)|preprint|Yes*|Qwen2.5-VL ViT|Qwen3-4B/8B|T, V|S, IR, QR, Nav|ReAct, HD|3A|
|SimpleDoc (2025)|EMNLP|Yes*|ColQwen-2.5|Multiple (Qwen)|T, V|J, IR, QR|-|1A, SR, SV|
|ViDoRAG (2025a)|EMNLP|Yes*|NV-Embed-V2, ColQwen2|Multiple (Qwen/LLaMa/GPT)|T, V|J, IR|ReAct, HD|3A, SR, SV|



Table 1: Survey of MP-VRDU models grouped by architecture. T = text; V = visual; L = layout. CoT = Chain of Thought; ReAct = Reason + Act; PE = Plan-Execute; QD = Question Decomposition; HD = Hierarchical / Coarse-to-Fine. S = Sparse; D = Dense; J = Joint; IR = Iterative Retrieval; QR = Query Reformulation; Nav = Navigation. _n_ A = _n_ agents; SR = Self-Refine; SV = Self-Verify; SC = Self-Consistency; SA = Sampling-Adjudication. OCR column: Yes _[∗]_ = not OCR-dependent; No _[∗]_ = OCRderived supervision explicitly used during training. ‘–’ indicates not reported. 

learn from cross-page evidence under long-context constraints (§5-§6). Finally, we consolidate MPVRDU datasets and evaluation practices, and identify key open challenges in data scarcity, benchmark contamination, scalability, grounding, reliability, and reproducible evaluation (§7-§8). 

cross-page evidence. The dataset catalogue covers datasets used by the surveyed systems, supplemented with a small number of relevant but underused MP-VRDU datasets. Full search terms and selection criteria are provided in Appendix A. 

## **3 MP-VRDU Architecture Overview** 

## **2 Methodology and Scope** 

This survey covers MP-VRDU systems published in the past 3 years across major venues such as CVPR, ACL, EMNLP and ICLR. Candidates were identified through keyword search terms such as “ _multi-page document_ ” and “ _long document understanding_ ” supplemented by forward-citation sweeps from anchor models and datasets. We manually screened candidates for substantive multipage contributions, such as novel architectural designs or training strategies that explicitly address 

The defining problem in MP-VRDU is evidence management: how a system manages cross-page evidence that is sparse, distributed, and frequently larger than any MLLM’s context window, so a system must decide what to retain, discard, or retrieve. These decisions are determined by architectural design, which fixes **how evidence is selected, preserved, and exposed for reasoning** , and we therefore organise the field by architecture rather than by the OCR dependency used in prior surveys (Ding et al., 2025). _Page-to-document_ 

2 

Figure 1: Illustration of MP-VRDU Architectures. Snowflake indicates component typically frozen; flame indicates component typically trained; both indicates even split across surveyed architectures. 

_encoding models_ jointly encode pages with crosspage representations (§3.1). _MLLM-centric adaptation models_ adapt MLLMs to multi-page inputs within context and token budgets (§3.2). _Retrievalaugmented pipelines_ retrieve relevant evidence before generation (§3.3). _Agentic pipelines_ iteratively inspect evidence using an LLM controller (§3.4). Reported performance across the four families on principal MP-VRDU benchmarks is provided in Appendix B.5 (Table 2). 

## **3.1 Page-to-Document Encoding Models** 

Page-to-document encoding models process each page through a shared layout-aware encoder such as LayoutLMv3 (Huang et al., 2022), that fuses OCR text, bounding-box layout, and visual features within the page, then connect per-page representations through a cross-page exchange mechanism in a single forward pass (Dong et al., 2024; Borchmann et al., 2025). By preserving page boundaries and cross-page relations within the architecture, this design provides a strong inductive bias for documents with regular structural templates. However, joint encoding caps total document length at the encoder’s capacity, and page-to-document compression discards fine-grained evidence as input grows, restricting this family’s applicability to documents of shorter length. 

following and reasoning capabilities. However, the backbone’s context length caps total document size, fine-grained signals depend on the fidelity of token compression (Jia et al., 2024), and document structure must be recovered implicitly from the input (Hannan et al., 2025). 

## **3.3 Retrieval-Augmented Pipelines** 

Retrieval-augmented pipelines decouple evidence selection from answer generation: a retriever first selects relevant evidence units, such as chunks, pages, or graph nodes, and the generator then produces an answer conditioned on this compact subset (Cho et al., 2024; Tanaka et al., 2025). This family spans single-pass retriever-generator systems (Cho et al., 2024) to multi-component pipelines that compose multiple LLM-instantiated “agents” in specialised roles (Han et al., 2025; Suri et al., 2025), all of which follow trajectories fixed at design time. As such, inference cost is bounded, and by passing only selected evidence to the generator, these pipelines scale to longer documents. However, evidence missed during retrieval is unrecoverable at generation, making retriever recall the binding constraint on this family, and upstream errors in chunking, OCR, or modality alignment propagate directly to the final answer. 

## **3.4 Adaptive-Trajectory Pipelines** 

## **3.2 MLLM-Centric Adaptation Models** 

MLLM-centric adaptation models process the document inside a pre-trained MLLM whose context window absorbs cross-page information directly, with adaptation ranging from training only lightweight modules to fine-tuning the backbone itself (Tanaka et al., 2024; Hu et al., 2025; Duan et al., 2025b; Xiong et al., 2026). This design benefits directly from the MLLM’s instruction- 

Adaptive-trajectory pipelines instantiate the agent paradigm of Plaat et al. and Wang et al., in which an LLM controller reasons, interacts with an external environment, and conditions subsequent actions on observed outcomes. Unlike the fixed “agents” of §3.3, the controller’s trajectory is not predetermined, rather, it selects each evidence-gathering action at inference time (Zheng et al., 2026; Sun et al., 2025), with actions surfacing as calls to ex- 

3 

ternal tools (Wang et al., 2025a; Yu et al., 2025) or as navigation over a pre-built document representation (Jain et al., 2025; Fu et al., 2026). ReAct (Yao et al., 2022) is the most common controller realisation (see §5.3), though iterative-reflection variants also appear. The cost of this flexibility is controllability: variable-length trajectories make inference cost hard to bound and controllers may commit early to wrong evidence with silent failure, but the trade-off is worthwhile, since unbounded cost is recoverable through engineering whereas missed evidence is not. 

## **4 Multi-Page Multimodal Representation** 

Every evidence-management decision is enacted over a specific modality, and its cost depends on how that modality is represented under the tension between page coverage and reading fidelity. This section reviews text (§4.1), visual (§4.2), and cross-page structural (§4.3) modelling under this pressure; fusion is deferred to Appendix B.1. 

## **4.1 Text Modality** 

Text in MP-VRDU presents four bottlenecks. **Volume** across pages frequently exceeds the MLLM context, addressed by aggressive compression into a fixed budget (Borchmann et al., 2025; Dong et al., 2024), by native long-context training that widens the input window (Duan et al., 2025b), or by indexing chunks and passing only selected text to the reader (Xie et al., 2024; Zhang et al., 2024); each shifts the lossy choice to a different stage. **Continuity** breaks when paragraphs, tables, and sections span page boundaries; it is preserved through document-level layers over perpage representations (Tito et al., 2023; Blau et al., 2024) or structure-aware chunking and adjacentpage expansion at retrieval time (Shin et al., 2025; Qian et al., 2026), each restoring local context at the cost of token-level detail or query relevance. **Noise** compounds as OCR errors accumulate across the document, dampened upstream through layoutaware pipelines or downstream by treating lowconfidence tokens as soft cues (Hannan et al., 2025), yet propagation through retrieval indexes remains uncharacterised. **Heterogeneity** in text density receives the least attention, with adaptive page selection (Li et al., 2025) the rare questionconditioned exception in a setting otherwise dominated by fixed-budget allocation. 

## **4.2 Visual Modality** 

Visual evidence preserves signals text extraction loses, but its computational cost introduces three bottlenecks. **Resolution** cannot be maintained at full document length, since token cost scales superlinearly under dynamic cropping (Zheng et al., 2026); it is managed by token-reduction modules that keep every page available at lower fidelity (Hu et al., 2025; Yu et al., 2024; Jia et al., 2024), thumbnail-based browsing with selective high-resolution inspection (Zheng et al., 2026; Jain et al., 2025), or high-fidelity reading of only retrieved pages that inherits the selector’s recall ceiling (Zhu et al., 2025a; Yan et al., 2025). **Multisignal competition** forces text-bearing pixels, nontextual evidence, and layout cues to share one per-page budget; visual RAG retains patch-level granularity but ranks by layout similarity rather than answer-region confirmation (Cho et al., 2024; Tanaka et al., 2025), element-level cropping preserves fine structure at the cost of detector dependence (Zhu et al., 2025a), and OCR-augmented visual reading delegates text recovery so the visual budget can prioritise non-textual evidence (Hannan et al., 2025). **Heterogeneity** from varying visual complexity remains underexplored; evidenceguided allocation (Yan et al., 2025) is one of the few question-conditioned approaches. 

## **4.3 Structure Modality** 

Structure turns isolated page observations into document evidence. Per-token layout remains relevant at MP scale (Zhu et al., 2025b), but the MPdistinctive difficulty is structural relations whose scope crosses pages. **Hierarchy** is the foundational problem, since sections, headings, and reading order organise the document as a whole rather than any individual page. Explicit parsers recover this as outlines or hierarchical indices (Shin et al., 2025; Sun et al., 2025; Fu et al., 2026), OCR-free backbones learn it end-to-end from page images (Hu et al., 2025), and per-segment layout tokens encode each region’s spatial arrangement without adding tokens to the context (Zhu et al., 2025b). **Cross-page references** demand resolving pointers such as “see Figure 3” to their referents on other pages, rather than ranking by surface similarity; graph- and map-based methods represent these references as edges between document elements (Wu et al., 2025b; Fu et al., 2026), at the cost of edge quality bounded by extraction heuristics. 

4 

**Spanning elements** such as multi-page tables and continued paragraphs require representations that bridge page boundaries without conflating distant content (Gong et al., 2025; Qian et al., 2026). 

## **5 Inference-Time Adaptation Strategies** 

Inference-time adaptation improves MP-VRDU performance by reshaping the inference pipeline around a frozen MLLM backbone rather than modifying its parameters. This makes it particularly suited to MP-VRDU, where supervised data is scarce and backbones must remain exchangeable across document types. We organise these strategies by the evidence-management decision they implement: selecting evidence (§5.1), reasoning over assembled evidence (§5.2), and iteratively acquiring further evidence (§5.3). 

## **5.1 Retrieval and Navigation** 

Retrieval narrows long documents to compact evidence units for downstream reasoning. Existing methods either rank evidences independently by query _similarity_ or traverse _relation-aware_ structures such as section hierarchies and crossreference graphs. 

**Similarity-based Retrieval.** Similarity-based retrieval scores evidences independently against the query and returns the top-ranked subset. Sparse lexical signals such as BM-25 serve as baselines or auxiliary channels, while dense retrieval in MP-VRDU typically uses vision-language lateinteraction encoders such as ColPali and ColQwen that align token-level query vectors with patchlevel page embeddings, preserving fine-grained alignment between query terms and page regions (Cho et al., 2024; Jain et al., 2025). Granularity varies from chunks to whole pages, with adaptive selectors adjusting the per-query retrieval count when evidence density is uneven (Li et al., 2025). A second reranking stage commonly refines an initial dense recall through LLM-based scoring over candidate evidences or page summaries, optionally under persistent working memory that carries cross-query context across iterations (Zhang et al., 2024; Xu et al., 2026). Independent scoring handles paraphrase and cross-modal queries well, but cannot follow cross-page references or multihop dependencies that no single passage carries. 

**Relation-aware Retrieval.** Relation-aware methods retrieve along explicit document structure 

rather than scoring evidences independently against the query. Tree-based variants extract section hierarchies and outlines, grouping evidences along semantic boundaries so retrieved chunks preserve parent context, with some routing queries through dual in-page and cross-page indices to recover both local and global evidence (Shin et al., 2025; Sun et al., 2025; Gong et al., 2025). Graph-based variants treat headings, entities, and cross-references as graph nodes with typed edges, allowing controllers to traverse from one passage to related ones rather than relying on similarity alone; extensions add logical relevance scoring or query-conditioned graph construction to bias traversal toward inferentially connected nodes (Wang et al., 2024c; Wu et al., 2025b; Zhang and Wu, 2026). Structurebased methods recover evidence links that similarity scoring misses, at the cost of dependence on parser quality and traversal cost that grows with the multi-hop count required to answer the question. 

## **5.2 Prompted Reasoning Strategies** 

Prompted reasoning strategies act over a single evidence buffer assembled prior to the reasoning call, which may be the full document representation, a retrieved subset, or evidence from an agentic step. We cover the four mainstream prompting paradigms; the coarse-to-fine pattern that recurs across many surveyed systems is treated separately in Appendix B.2. 

**Sequential Reasoning.** A single reasoning thread proceeds linearly, optionally with revision. _Chain-of-Thought (CoT)_ (Wei et al., 2022) is widely adopted in MP-VRDU to bind evidence from different pages step by step and decompose multi-hop questions into single-hop steps (Jia et al., 2024; Xu et al., 2026), but operates only over the assembled evidence, locating its contribution in the fidelity rather than the recall stage of the pipeline. _Self-Reflection_ (Reflexion-style) (Shinn et al., 2023) extends the single thread with iterative revision against a verifier or critic (Jain et al., 2025; Wang et al., 2025a), reducing single-pass error at the cost of additional rounds and a self-evaluation signal whose reliability bounds the gains. 

**Parallel Exploration.** Multiple reasoning paths are produced in parallel and combined into a single answer. _Self-Consistency_ (Wang et al., 2022) samples multiple chains and selects the majority answer, with sampling-adjudication variants replacing the vote with an LLM judge (Zhu et al., 2025a), 

5 

scaling robustness with inference cost. _Tree-ofThoughts (ToT)_ (Yao et al., 2023) explores reasoning branches with backtracking, but has seen limited uptake in MP-VRDU, where evidence sparsity rather than search-space branching is the dominant difficulty. 

## **5.3 Agentic Methods** 

Agentic methods distribute the MP-VRDU workflow across one or more LLM-instantiated controllers, either iterating evidence acquisition under inference-time control or coordinating across specialised roles. We note that _agentic_ is used ambiguously in the MP-VRDU literature, covering both systems whose controller selects actions at inference time, following the framing by Plaat et al. and Wang et al., and systems composed of multiple LLM-instantiated agents in fixed roles; we treat both senses here, following the architectural distinction in §3.3–§3.4. 

**ReAct and Tool-Augmented Reasoning.** ReAct prompting (Yao et al., 2022) augments the action space with language thoughts, interleaving thoughtaction emissions in a single stream whose observations feed the next step, letting a single controller acquire evidence during inference rather than reasoning over a fixed buffer. The action space takes two forms in MP-VRDU. Document-side tool calls invoke external retrievers, parsers, or OCR engines, with each round concentrating on a chosen page or region (Sun et al., 2025; Wu et al., 2025a), lifting the single-step recall ceiling of retrievalaugmented pipelines. Structured navigation instead operates over a pre-built document representation (see §4.3), with the controller selecting which node, section, or page to inspect next (Zheng et al., 2026). Adaptive-trajectory systems that lack the thoughtaction schema, such as those built on Reflexionstyle iterative refinement (Jain et al., 2025; Fu et al., 2026), share the iterative control regime without being ReAct in the strict sense. 

**Multi-Agent Orchestration.** Multi-agent orchestration distributes the workflow across multiple controllers whose interaction pattern, rather than a single monolithic controller, drives evidence gathering and answer synthesis. Each controller operates on a narrower input and clearer objective than a single backbone tasked with the entire workflow, mitigating the cognitive overload that arises when one model must locate, read, verify, and answer over the entire document (Yu et al., 2025). Two patterns 

recur. _Modality-specialised_ decomposition assigns separate controllers to text and visual evidence and adds a synthesis controller that reconciles their outputs, preserving single-modality reading fidelity at the cost of a reconciliation step that lacks access to the original document (Han et al., 2025; Duan et al., 2025a). _Role-specialised_ decomposition assigns controllers to procedural stages such as planning, execution, verification, and synthesis, with routing between stages when a downstream controller detects an error (Yu et al., 2025; Sun et al., 2025). Separating judgement from generation reduces the cognitive blind spots of self-correction, but introduces inter-controller compatibility costs and a recall ceiling bounded by upstream stages. 

## **6 Training Strategies** 

Training internalises the evidence-management decisions of §5 into model parameters, supplying capabilities that prompting cannot: cross-page evidence selection (§6.1), document-scale reasoning over assembled evidence (§6.2), and reliable agentic acquisition (§6.3). Two cross-cutting techniques recur across all three (§6.4). 

## **6.1 Trained Retrieval Components** 

Off-the-shelf encoders such as ColPali and ColQwen (Faysse et al., 2024a) score each page independently, leaving retrieval dominated by local query-page similarity rather than documentlevel context. Retriever training has therefore explored three complementary directions. **Modalityalignment** methods align visual page embeddings with OCR-derived text representations, enabling page-image retrieval without late-interaction scoring at inference (Tanaka et al., 2025), trading index size for query-time efficiency. **Cross-pagefusion** methods enrich each page representation with neighbouring-page context (Qian et al., 2026), recovering distributed evidence without rebuilding the encoder, though the optimal neighbour window is corpus-dependent and rarely tuned per query. **Logical-relevance** methods train graphbased scorers that rank pages by their position in cross-page reasoning paths rather than embedding similarity (Wu et al., 2025b), addressing multi-hop questions whose evidence pages are individually low-scoring at the cost of inheriting the graphextractor’s failure modes. The three directions are not interchangeable: modality-alignment improves throughput, cross-page-fusion improves recall on 

6 

adjacent evidence, and logical-relevance improves recall on inferentially-connected evidence. 

## **6.2 Trained Reasoning Methods** 

Prompted reasoning leaves quality sensitive to prompt design and base-model behaviour, motivating training that internalises reasoning patterns into the backbone. **Trace-imitation** variants supervise the backbone on demonstration chains wrapped in fixed schema tags such as <think>, <evidence>, and <answer> (Yan et al., 2025; Guo et al., 2026), which provides structure that downstream verifiers can audit but inherits any biases or omissions in the teacher’s chains. **Reward-driven** variants optimise the reasoning policy with Group Relative Policy Optimisation (GRPO) against rewards combining answer accuracy, evidence-page recall, and format compliance (Xiong et al., 2026; Xu et al., 2026), typically warm-started from a traceimitation stage; the two are therefore sequential rather than alternative, with reward-driven training refining what imitation alone cannot. **Uncertaintycalibration** variants learn an aggregation head that selects across candidate outputs by calibrated uncertainty (Hannan et al., 2025), providing a learned self-consistency signal without test-time chain sampling, which suits parameter or compute budgets that preclude sampling-based approaches. 

## **6.3 Trained Agentic Methods** 

Prompt-only agentic control is unreliable on smaller open-weight backbones, which cannot consistently follow ReAct schemas and select among tool calls. **Trajectory-distillation** variants clone agentic trajectories from a stronger teacher through SFT on think-action-observation turns (Zheng et al., 2026; Zhang et al., 2026), with closed-loop teacher collection essential because retrieval and fetching change subsequent observations; distillation is bounded by teacher quality and inherits the teacher’s blind spots. **Reinforcement-learned** variants optimise the agentic policy directly against trajectory-level rewards combining answer correctness, evidence recall, and tool-use shape (Wang et al., 2026; Lin et al., 2026), lifting the teacherquality ceiling at the cost of training instability that multi-turn variants address through entropybased stopping, similarity-weighted baselines, or per-role budget allocation. The surveyed evidence does not cleanly separate the two, since reported comparisons confound the training objective with backbone, reward design, and trajectory budget. 

## **6.4 Cross-Cutting Training Techniques** 

Two techniques recur across retriever, reasoner, and agent training. **Parameter-efficient tuning** (PEFT) is dominant: full backbone fine-tuning is prohibitive at long-document context lengths and risks erasing pretrained capabilities, so the prevailing pattern freezes the backbone and trains lightweight modules through Low-Rank Adaptation (LoRA), projection heads, or task-specific adapters (Zhang et al., 2024; Qian et al., 2026; Zhang et al., 2026), with the choice of module driven more by compute budget than by target component. **Multi-stage curriculum training** addresses a different concern: several systems start on single-page objectives before introducing document-scale supervision (Hu et al., 2025; Hannan et al., 2025; Xiong et al., 2026), exploiting the fact that text reading and layout understanding transfer from single-page data while cross-page integration must be learned separately. Their recurrence across dissimilar systems indicates that cross-page integration does not emerge from single-page supervision alone. 

## **7 Datasets** 

Multi-page documents impose a structural constraint absent from the single-page setting: perpage annotation does not scale, and multi-page corpora are scarce because long PDFs are harder to source, license, and parse than single pages. This constraint shapes all three dataset roles in MPVRDU differently from SP-VRDU. **Pretraining** is the exception rather than the norm, confined to a few page-to-document and MLLM-centric systems that perform continued pretraining on existing backbones, since multi-page pretraining corpora remain small relative to the single-page archives that preceded them. **Fine-tuning** has converged on LVLM-synthesised supervision over scraped PDFs, with reinforcement learning increasingly layered on top to shape evidence-grounding and trajectory behaviour rather than to scale the data itself. **Benchmarking** has stratified into three tiers: single-page extractive baselines, mid-length multipage suites, and long-document multimodal benchmarks requiring cross-page integration, with retrieval and page-localisation increasingly scored alongside answer accuracy. The same scarcity of source documents that drives synthetic generation also concentrates fine-tuning and evaluation onto the same small pool of PDFs, a contamination risk we return to in §8. For full details see Appendix C. 

7 

## **8 Challenges and Future Work** 

A consistent pattern emerges across MP-VRDU systems: progress comes from moving evidence handling out of the backbone, through compression, retrieval, or iterative orchestration, but each shift relocates the bottleneck rather than removing it. Five challenges define the open frontier: three concern system capability (the evidence pipeline itself, faithfulness, and efficiency) and two concern data and evaluation (supervision integrity and evaluation scope). 

**Evidence Pipeline Bottleneck.** MP-VRDU systems must reconcile high-resolution per-page reading with document-scale evidence aggregation, an objective that exceeds any MLLM’s compute and context budget (Ma et al., 2024). Every architectural family takes a lossy compromise: compression silently drops fine-grained content and is typically fixed before the question is seen; retrieval bounds answer accuracy from above while broader retrieval introduces attention dilution; and iterative retrieval lifts this ceiling only partially, inheriting the recall of every tool it calls. Cross-page structure compounds this further, as none of the routes in §4.3 scales cleanly. Future work should explore query-conditional compression, retrieval that follows cross-references and logical relations, and learned cross-page structural representations. 

**Faithfulness and Abstention.** Long documents amplify hallucination risk; when retrieval misses or compression discards the answer-bearing region, current systems still produce confident answers. Few surveyed systems support abstention, calibrated uncertainty, or citation-grounded generation, leaving downstream users without signals to distinguish supported answers from fabrications. Future work should integrate evidence-grounded decoding, learned abstention against unanswerable or unretrievable questions, and trajectory-level faithfulness audits that verify the cited evidence supports the final answer. 

**Efficiency and Deployability.** Agentic and retrieval-augmented pipelines routinely issue tens of LLM calls per question, with token costs scaling jointly with document length and trajectory length. The surveyed literature reports almost no inferencecost analysis: end-to-end latency, token budgets per query, and accuracy–cost Pareto curves are largely absent, despite their centrality to enterprise deployment. Future work should treat compute as 

a first-class evaluation axis, develop efficient inference techniques specific to long documents such as cross-round caching, speculative decoding, and adaptive depth, and explore compact MP-VRDU models (Hannan et al., 2025) that operate under realistic on-device or privacy constraints. 

**Supervision Scarcity and Contamination.** Perpage annotation does not scale to documents averaging tens of pages, so most systems rely on LVLM-synthesised QA over scraped PDFs (Tanaka et al., 2024; Jain et al., 2025; Xiong et al., 2026; Zhang et al., 2026). Compounding this, benchmark construction often reuses documents from earlier suites: MMLongBench-Doc draws documents from DUDE, SlideVQA, and ChartQA (Ma et al., 2024), while MP-DocVQA is built on DocVQA (Tito et al., 2023; Mathew et al., 2021). Together, MLLM-generated QA data (Deng et al., 2025) and benchmark-reused fine-tuning sets (Xiong et al., 2026; Duan et al., 2025b) risk contamination: models may train on benchmarkderived documents and be evaluated on overlapping data, sometimes judged by the same MLLM family. Future work should adopt held-out documents, declare cross-suite overlap, and separate generators from evaluators. More details and overlap matrix available in Appendix C.4. 

**Evaluation Methodology and Scope.** Beyond contamination, two further evaluation gaps persist. Answer-only accuracy masks retrieval failure and rewards correct answers reached through incorrect reasoning; newer benchmarks score retrieval and page-localisation alongside VQA-style answers (Hui et al., 2024; Xiong et al., 2026; Zhu et al., 2025a), but the practice is not yet standardised, and agentic pipelines produce stochastic trajectories that further complicate reproducibility. Task, language, and domain scope also remain narrow. Real-world MP-VRDU spans summarisation and structured extraction beyond VQA, multilingual layouts beyond English, and domainspecialised settings such as medical, legal, and financial documents, as well as streaming inputs that arrive page-by-page. These dimensions are inherited from SP-VRDU (Ding et al., 2025) rather than introduced by the multi-page setting and remain underexplored in both regimes. Future work should therefore standardise trajectory-level evaluation with retrieval and grounding metrics, report stochasticity bounds, and broaden coverage along these inherited dimensions. 

8 

## **Limitations** 

This survey covers multi-page VRDU models published before May 2026 and is bounded by the rapid pace of the field, with new architectures, training-free methods, and benchmarks appearing during preparation that may not be reflected here. Selection prioritises representative coverage of architectural and inference-time strategies over exhaustive enumeration. Cross-model performance comparisons are inherently limited since training distributions, benchmark coverage, and evaluation metrics are bespoke per system, and quantitative claims should be interpreted as indicative rather than definitive. 

## **References** 

- Ali Furkan Biten, Rubèn Tito, Lluis Gomez, Ernest Valveny, and Dimosthenis Karatzas. 2022. OCRIDL: OCR annotations for industry document library dataset. In _European Conference on Computer Vision Workshops (Text in Everything)_ . 

- Tsachi Blau, Sharon Fogel, Roi Ronen, Alona Golts, Roy Ganz, Elad Ben Avraham, Aviad Aberdam, Shahar Tsiper, and Ron Litman. 2024. Gram: Global reasoning for multi-page vqa. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition_ , pages 15598–15607. 

- Łukasz Borchmann, Michał Pietruszka, Wojciech Ja´skowski, Dawid Jurkiewicz, Piotr Halama, Paweł Józiak, Łukasz Garncarek, Paweł Liskowski, Karolina Szyndler, Andrzej Gretkowski, and 1 others. 2025. Arctic-tilt. business document understanding at sub-billion scale. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 6: Industry Track)_ , pages 264– 283. 

- Jian Chen, Ruiyi Zhang, Yufan Zhou, Tong Yu, Franck Dernoncourt, Jiuxiang Gu, Ryan Rossi, Changyou Chen, and Tong Sun. 2025. Sv-rag: Loracontextualizing adaptation of mllms for long document understanding. In _The Thirteenth International Conference on Learning Representations_ . 

- Wenhu Chen, Hongmin Wang, Jianshu Chen, Yunkai Zhang, Hong Wang, Shiyang Li, Xiyou Zhou, and William Yang Wang. 2020. TabFact: A large-scale dataset for table-based fact verification. In _International Conference on Learning Representations (ICLR)_ . 

- Yew Ken Chia, Liying Cheng, Hou Pong Chan, Maojia Song, Chaoqun Liu, Mahani Aljunied, Soujanya Poria, and Lidong Bing. 2025. M-longdoc: A benchmark for multimodal super-long document understanding and a retrieval-aware tuning framework. 

In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 9244–9261. 

- Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2024. M3docrag: Multimodal retrieval is what you need for multi-page multi-document understanding. _arXiv preprint arXiv:2411.04952_ . 

- Jaemin Cho, Debanjan Mahata, Ozan Irsoy, Yujie He, and Mohit Bansal. 2025. M3docvqa: Multi-modal multi-page multi-document understanding. In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 6178–6188. 

- Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A. Smith, and Matt Gardner. 2021. A dataset of information-seeking questions and answers anchored in research papers. In _Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL-HLT)_ , pages 4599–4610. Association for Computational Linguistics. 

- Chao Deng, Jiale Yuan, Pi Bu, Peijie Wang, ZhongZhi Li, Jian Xu, Xiao-Hui Li, Yuan Gao, Jun Song, Bo Zheng, and 1 others. 2025. Longdocurl: a comprehensive multimodal long document benchmark integrating understanding, reasoning, and locating. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 1135–1159. 

- Yihao Ding, Soyeon Caren Han, Jean Lee, and Eduard Hovy. 2026. Deep learning based visually rich document content understanding: A survey. _Artificial Intelligence Review_ . 

- Yihao Ding, Siwen Luo, Hyunsuk Chung, and Soyeon Caren Han. 2023. Vqa: A new dataset for real-world vqa on pdf documents. In _Joint European Conference on Machine Learning and Knowledge Discovery in Databases_ , pages 585–601. Springer. 

- Yihao Ding, Siwen Luo, Yue Dai, Yanbei Jiang, Zechuan Li, Geoffrey Martin, and Yifan Peng. 2025. A survey on mllm-based visually rich document understanding: Methods, challenges, and emerging trends. _arXiv preprint arXiv:2507.09861_ . 

- Yihao Ding, Kaixuan Ren, Jiabin Huang, Siwen Luo, and Soyeon Caren Han. 2024. Mmvqa: A comprehensive dataset for investigating multipage multimodal information retrieval in pdf-based visual question answering. In _IJCAI_ , pages 6243–6251. 

- Kuicai Dong, Yujing Chang, Derrick Goh Xin Deik, Dexun Li, Ruiming Tang, and Yong Liu. 2025. Mmdocir: Benchmarking multimodal retrieval for long documents. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 30959–30993. 

9 

- Qi Dong, Lei Kang, and Dimosthenis Karatzas. 2024. Multi-page document vqa with recurrent memory transformer. In _International Workshop on Document Analysis Systems_ , pages 57–70. Springer. 

- Tongtong Duan, Minghao Hu, Liang Xue, Chunming Liu, Guotong Geng, Wei Luo, and Zhunchen Luo. 2025a. M2rag: A multi-agent and multimodal fusion framework for retrieval-augmented document qa. In _International Conference on Neural Information Processing_ , pages 507–520. Springer. 

- Yuchen Duan, Zhe Chen, Yusong Hu, Weiyun Wang, Shenglong Ye, Botian Shi, Lewei Lu, Qibin Hou, Tong Lu, Hongsheng Li, and 1 others. 2025b. Docopilot: Improving multimodal models for documentlevel understanding. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 4026–4037. 

- Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024a. ColPali: Efficient document retrieval with vision language models. 

- Manuel Faysse, Hugues Sibille, Tony Wu, Bilel Omrani, Gautier Viaud, Céline Hudelot, and Pierre Colombo. 2024b. Colpali: Efficient document retrieval with vision language models. _arXiv preprint arXiv:2407.01449_ . 

- ShunLiang Fu, Yanxin Zhang, Yixin Xiang, Xiaoyu Du, and Jinhui Tang. 2026. Dmap: Human-aligned structural document map for multimodal document understanding. In _Proceedings of the ACM Web Conference 2026_ , pages 2037–2048. 

- Sensen Gao, Shanshan Zhao, Xu Jiang, Lunhao Duan, Yong Xien Chng, Qing-Guo Chen, Weihua Luo, Kaifu Zhang, Jia-Wang Bian, and Mingming Gong. 2025. Scaling beyond context: A survey of multimodal retrieval-augmented generation for document understanding. _arXiv preprint arXiv:2510.15253_ . 

- Hamza Gbada, Karim Kalti, and Mohamed Ali Mahjoub. 2025. Deep learning approaches for information extraction from visually rich documents: datasets, challenges and methods. _International Journal on Document Analysis and Recognition_ , 28(1):121–142. 

- Ziyu Gong, Chengcheng Mai, and Yihua Huang. 2025. Mhier-rag: Multi-modal rag for visualrich document question-answering via hierarchical and multi-granularity reasoning. _arXiv preprint arXiv:2508.00579_ . 

- Jindi Guo, Chaozheng Huang, Haoyi Tao, Zhulin An, Guolin Ke, and Xi Fang. 2026. End-to-end document understanding via chain-of-reading. 

- Siwei Han, Peng Xia, Ruiyi Zhang, Tong Sun, Yun Li, Hongtu Zhu, and Huaxiu Yao. 2025. Mdocagent: A multi-modal multi-agent framework for document understanding. _arXiv preprint arXiv:2503.13964_ . 

- Tanveer Hannan, Dimitrios Mallios, Parth Pathak, Faegheh Sardari, Thomas Seidl, Gedas Bertasius, Mohsen Fayyaz, and Sunando Sengupta. 2025. Docslm: A small vision-language model for long multimodal document understanding. _arXiv preprint arXiv:2511.11313_ . 

- Anwen Hu, Haiyang Xu, Jiabo Ye, Ming Yan, Liang Zhang, Bo Zhang, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. 2024. mPLUG-DocOwl 1.5: Unified structure learning for OCR-free document understanding. In _Findings of the Association for Computational Linguistics: EMNLP 2024_ , pages 3096–3120. 

- Anwen Hu, Haiyang Xu, Liang Zhang, Jiabo Ye, Ming Yan, Ji Zhang, Qin Jin, Fei Huang, and Jingren Zhou. 2025. mplug-docowl2: High-resolution compressing for ocr-free multi-page document understanding. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 5817–5834. 

- Yupan Huang, Tengchao Lv, Lei Cui, Yutong Lu, and Furu Wei. 2022. Layoutlmv3: Pre-training for document ai with unified text and image masking. In _Proceedings of the 30th ACM international conference on multimedia_ , pages 4083–4091. 

- Yulong Hui, Yao Lu, and Huanchen Zhang. 2024. Uda: A benchmark suite for retrieval augmented generation in real-world document analysis. _Advances in Neural Information Processing Systems_ , 37:67200–67217. 

- Chelsi Jain, Yiran Wu, Yifan Zeng, Jiale Liu, Shengyu Dai, Zhenwen Shao, Qingyun Wu, and Huazheng Wang. 2025. Simpledoc: Multi-modal document understanding with dual-cue page retrieval and iterative refinement. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 28398–28415. 

- Mengzhao Jia, Wenhao Yu, Kaixin Ma, Tianqing Fang, Zhihan Zhang, Siru Ouyang, Hongming Zhang, Dong Yu, and Meng Jiang. 2024. Leopard: A vision language model for text-rich multi-image tasks. _arXiv preprint arXiv:2410.01744_ . 

- Lei Kang, Rubèn Tito, Ernest Valveny, and Dimosthenis Karatzas. 2024. Multi-page document visual question answering using self-attention scoring mechanism. In _International Conference on Document Analysis and Recognition_ , pages 219–232. Springer. 

- David Lewis, Gady Agam, Shlomo Argamon, Ophir Frieder, David Grossman, and Jefferson Heard. 2006. Building a test collection for complex document information processing. In _Proceedings of the 29th annual international ACM SIGIR conference on Research and development in information retrieval_ , pages 665–666. 

- Zongmin Li, Yachuan Li, Lei Kang, Dimosthenis Karatzas, and Wenkang Ma. 2025. Avir: Adaptive visual in-document retrieval for efficient multi-page document question answering. In _Proceedings of the 7th ACM International Conference on Multimedia in Asia_ , pages 1–7. 

10 

- Jiahang Lin, Kai Hu, Binghai Wang, Yuhao Zhou, Zhiheng Xi, Honglin Guo, Shichun Liu, Junzhe Wang, Shihan Dou, Enyu Zhou, Hang Yan, Zhenhua Han, Tao Gui, Qi Zhang, and Xuanjing Huang. 2026. Mmdoc-r1: Training agents for long document visual question answering through multi-turn reinforcement learning. _arXiv preprint arXiv:2604.13579_ . 

- Keliang Liu, Zizhi Chen, Mingcheng Li, Jingqun Tang, Dingkang Yang, and Lihua Zhang. 2025. Resolving evidence sparsity: Agentic context engineering for long-document understanding. _arXiv preprint arXiv:2511.22850_ . 

- Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. _Transactions of the association for computational linguistics_ , 12:157–173. 

- Eric López, Artemis Llabrés, and Ernest Valveny. 2025. Enhancing document vqa models via retrievalaugmented generation. In _International Conference on Document Analysis and Recognition_ , pages 92– 107. Springer. 

- Yubo Ma, Yuhang Zang, Liangyu Chen, Meiqi Chen, Yizhu Jiao, Xinze Li, Xinyuan Lu, Ziyu Liu, Yan Ma, Xiaoyi Dong, and 1 others. 2024. Mmlongbench-doc: Benchmarking long-context document understanding with visualizations. _Advances in Neural Information Processing Systems_ , 37:95963–96010. 

- Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In _Findings of the association for computational linguistics: ACL 2022_ , pages 2263– 2279. 

- Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In _Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision_ , pages 1697–1706. 

- Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In _Proceedings of the IEEE/CVF winter conference on applications of computer vision_ , pages 2200–2209. 

- Linyong Nan, Chiachun Hsieh, Ziming Mao, Xi Victoria Lin, Neha Verma, Rui Zhang, Wojciech Kry´sci´nski, Hailey Schoelkopf, Riley Kong, Xiangru Tang, Mutethia Mutuma, Ben Rosand, Isabel Trindade, Renusree Bandaru, Jacob Cunningham, Caiming Xiong, and Dragomir Radev. 2022. FeTaQA: Freeform table question answering. _Transactions of the Association for Computational Linguistics_ , 10:35–49. 

- Panupong Pasupat and Percy Liang. 2015. Compositional semantic parsing on semi-structured tables. In _Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language_ 

_Processing (Volume 1: Long Papers)_ , pages 1470– 1480. Association for Computational Linguistics. 

- Aske Plaat, Max van Duijn, Niki Van Stein, Mike Preuss, Peter van der Putten, and Kees Joost Batenburg. 2025. Agentic large language models, a survey. _Journal of Artificial Intelligence Research_ , 84. 

- Le Qi, Shangwen Lv, Hongyu Li, Jing Liu, Yu Zhang, Qiaoqiao She, Hua Wu, Haifeng Wang, and Ting Liu. 2022. Dureadervis: A chinese dataset for opendomain document visual question answering. In _Findings of the Association for Computational Linguistics: ACL 2022_ , pages 1338–1351. 

- Bing Qian, Kaiwei Deng, Yuexin Wu, Jianming Zhang, Juanjuan Sun, Yanru Xue, and Chunxiao Fan. 2026. Accurate multi-page document retrieval by effectively fusing context information across pages. _Electronics_ , 15(7):1353. 

- Yongxin Shi, Jiapeng Wang, Zeyu Shan, Dezhi Peng, Zening Lin, and Lianwen Jin. 2026. Urag: Unified retrieval and generation in multimodal llms for efficient long document understanding. In _Proceedings of the AAAI Conference on Artificial Intelligence_ . 

- Joongmin Shin, Chanjun Park, Jeongbae Park, Jaehyung Seo, and Heui-Seok Lim. 2025. Multidocfusion: Hierarchical and multimodal chunking pipeline for enhanced rag on long industrial documents. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 20996–21015. 

- Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. _Advances in neural information processing systems_ , 36:8634–8652. 

- Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In _Proceedings of the IEEE/CVF conference on computer vision and pattern recognition_ , pages 8317–8326. 

- Tomasz Stanisławek, Filip Grali´nski, Anna Wróblewska, Dawid Lipi´nski, Agnieszka Kaliska, Paulina Rosalska, Bartosz Topolski, and Przemysław Biecek. 2021. Kleister: Key information extraction datasets involving long documents with complex layouts. In _Document Analysis and Recognition – ICDAR 2021_ , pages 564–579. Springer. 

- Nishant Subramani, Alexandre Matton, Malcolm Greaves, and Adrian Lam. 2020. A survey of deep learning approaches for ocr and document understanding. _arXiv preprint arXiv:2011.13534_ . 

- Li Sun, Liu He, Shuyue Jia, Yangfan He, and Chenyu You. 2025. Docagent: An agentic framework for multi-modal long-context document understanding. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 17712–17727. 

11 

- Manan Suri, Puneet Mathur, Kanika Goswami, Ryan A Rossi, Franck Dernoncourt, and Dinesh Manocha. 2025. Visdom: Multi-document qa with visually rich elements using multimodal retrieval-augmented generation. In _Proceedings of the 2025 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ . 

- Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. 2025. Vdocrag: Retrieval-augmented generation over visually-rich documents. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 24827–24837. 

- Ryota Tanaka, Taichi Iki, Kyosuke Nishida, Kuniko Saito, and Jun Suzuki. 2024. Instructdoc: A dataset for zero-shot generalization of visual document understanding with instructions. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 38, pages 19071–19079. 

- Ryota Tanaka, Kyosuke Nishida, Kosuke Nishida, Taku Hasegawa, Itsumi Saito, and Kuniko Saito. 2023. Slidevqa: A dataset for document visual question answering on multiple images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 37, pages 13636–13645. 

- Ryota Tanaka, Kyosuke Nishida, and Sen Yoshida. 2021. Visualmrc: Machine reading comprehension on document images. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 35, pages 13878– 13888. 

- Rubèn Tito, Dimosthenis Karatzas, and Ernest Valveny. 2023. Hierarchical multimodal transformers for multipage docvqa. _Pattern Recognition_ , 144:109834. 

- Jordy Van Landeghem, Rubèn Tito, Łukasz Borchmann, Michał Pietruszka, Pawel Joziak, Rafal Powalski, Dawid Jurkiewicz, Mickaël Coustaty, Bertrand Anckaert, Ernest Valveny, and 1 others. 2023. Document understanding dataset and evaluation (dude). In _Proceedings of the IEEE/CVF International Conference on Computer Vision_ , pages 19528–19540. 

- Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, and 1 others. 2024a. A survey on large language model based autonomous agents. _Frontiers of Computer Science_ , 18(6):186345. 

- Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, and Feng Zhao. 2025a. Vidorag: Visual document retrieval-augmented generation via dynamic iterative reasoning agents. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 9124– 9145. 

- Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, and Feng Zhao. 2026. Vrag-rl: Empower visionperception-based rag for visually rich information 

understanding via iterative reasoning with reinforcement learning. _Advances in Neural Information Processing Systems_ , 38:57133–57160. 

- Weiyun Wang, Shuibo Zhang, Yiming Ren, Yuchen Duan, Tiantong Li, Shuo Liu, Mengkang Hu, Zhe Chen, Kaipeng Zhang, Lewei Lu, and 1 others. 2024b. Needle in a multimodal haystack. _Advances in Neural Information Processing Systems_ , 37:20540– 20565. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc Le, Ed Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2022. Self-consistency improves chain of thought reasoning in language models. _arXiv preprint arXiv:2203.11171_ . 

- Yu Wang, Nedim Lipka, Ryan A Rossi, Alexa Siu, Ruiyi Zhang, and Tyler Derr. 2024c. Knowledge graph prompting for multi-document question answering. In _Proceedings of the AAAI conference on artificial intelligence_ , volume 38, pages 19206–19214. 

- Zhaowei Wang, Wenhao Yu, Xiyu Ren, Jipeng Zhang, Yu Zhao, Rohit Saxena, Liang Cheng, Ginny Wong, Simon See, Pasquale Minervini, Yangqiu Song, and Mark Steedman. 2025b. MMLongBench: Benchmarking long-context vision-language models effectively and thoroughly. In _Advances in Neural Information Processing Systems_ . 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, and 1 others. 2022. Chain-of-thought prompting elicits reasoning in large language models. _Advances in neural information processing systems_ , 35:24824– 24837. 

- Junda Wu, Yu Xia, Tong Yu, Xiang Chen, Sai Sree Harsha, Akash V Maharaj, Ruiyi Zhang, Victor Bursztyn, Sungchul Kim, Ryan A Rossi, and 1 others. 2025a. Doc-react: Multi-page heterogeneous document question-answering. In _Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)_ , pages 67–78. 

- Xixi Wu, Yanchao Tan, Nan Hou, Ruiyang Zhang, and Hong Cheng. 2025b. Molorag: Bootstrapping document understanding via multi-modal logic-aware retrieval. In _Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing_ , pages 14035–14056. 

- Renqiu Xia, Song Mao, Xiangchao Yan, Hongbin Zhou, Bo Zhang, Haoyang Peng, Jiahao Pi, Daocheng Fu, Wenjie Wu, Hancheng Ye, and 1 others. 2024. Docgenome: An open large-scale scientific document benchmark for training and testing multi-modal large language models. _arXiv preprint arXiv:2406.11633_ . 

- Biao Xiang, Soyeon Caren Han, and Yihao Ding. 2026. Bridge: Benchmark for multi-hop reasoning in long multimodal documents with grounded evidence. _arXiv preprint arXiv:2603.07931_ . 

12 

- Xudong Xie, Hao Yan, Liang Yin, Yang Liu, Jing Ding, Minghui Liao, Yuliang Liu, Wei Chen, and Xiang Bai. 2024. Wukong: A large multimodal model for efficient long pdf reading with end-to-end sparse sampling. _arXiv preprint arXiv:2410.05970_ . 

- Junyu Xiong, Yonghui Wang, Weichao Zhao, Chenyu Liu, Bing Yin, Wengang Zhou, and Houqiang Li. 2026. Docr1: Evidence page-guided grpo for multipage document understanding. In _Proceedings of the AAAI Conference on Artificial Intelligence_ , volume 40, pages 11178–11186. 

- Mingjun Xu, Jinhan Dong, Jue Hou, Zehui Wang, Sihang Li, Zhifeng Gao, Renxin Zhong, and Hengxing Cai. 2026. Mm-r5: Multimodal reasoning-enhanced reranker via reinforcement learning for document retrieval. In _Proceedings of the AAAI Conference on Artificial Intelligence_ . 

- Hao Yan, Yuliang Liu, Xingchen Liu, Yuyi Zhang, Minghui Liao, Jihao Wu, Wei Chen, and Xiang Bai. 2025. Docseeker: Structured visual reasoning with evidence grounding for long document understanding. _arXiv preprint_ . 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023. Tree of thoughts: Deliberate problem solving with large language models. _Advances in neural information processing systems_ , 36:11809–11822. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2022. React: Synergizing reasoning and acting in language models. _arXiv preprint arXiv:2210.03629_ . 

- Xinlei Yu, Chengming Xu, Zhangquan Chen, Yudong Zhang, Shilin Lu, Cheng Yang, Jiangning Zhang, Shuicheng Yan, and Xiaobin Hu. 2025. Visual document understanding and reasoning: A multi-agent collaboration framework with agent-wise adaptive test-time scaling. _arXiv preprint arXiv:2508.03404_ . 

- Ya-Qi Yu, Minghui Liao, Jiwen Zhang, and Jihao Wu. 2024. Texthawk2: A large vision-language model excels in bilingual ocr and grounding with 16x fewer tokens. _arXiv preprint arXiv:2410.05261_ . 

   - Yongyue Zhang and Yaxiong Wu. 2026. Mldocrag: Multimodal long-context document retrieval augmented generation. _arXiv preprint arXiv:2602.10271_ . 

   - Yilun Zhao, Yunxiang Li, Chenying Li, and Rui Zhang. 2022. MultiHiertt: Numerical reasoning over multi hierarchical tabular and textual data. In _Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)_ , pages 6588–6600. Association for Computational Linguistics. 

   - Yuanlei Zheng, Pei Fu, Hang Li, Ziyang Wang, Yuyi Zhang, Wenyu Ruan, Xiaojin Zhang, Zhongyu Wei, Zhenbo Luo, Jian Luan, and 1 others. 2026. Doc-v*: Coarse-to-fine interactive visual reasoning for multi-page document vqa. _arXiv preprint arXiv:2604.13731_ . 

   - Dawei Zhu, Rui Meng, Jiefeng Chen, Sujian Li, Tomas Pfister, and Jinsung Yoon. 2025a. Doclens: A tool-augmented multi-agent framework for long visual document understanding. _arXiv preprint arXiv:2511.11552_ . 

   - Fengbin Zhu, Wenqiang Lei, Fuli Feng, Chao Wang, Haozhou Zhang, and Tat-Seng Chua. 2022. Towards complex document understanding by discrete reasoning. In _Proceedings of the 30th ACM International Conference on Multimedia_ , pages 4857–4866. 

   - Zhaoqing Zhu, Chuwei Luo, Zirui Shao, Feiyu Gao, Hangdi Xing, Qi Zheng, and Ji Zhang. 2025b. A simple yet effective layout token in large language models for document understanding. In _Proceedings of the Computer Vision and Pattern Recognition Conference_ , pages 14472–14482. 

   - Anni Zou, Wenhao Yu, Hongming Zhang, Kaixin Ma, Deng Cai, Zhuosheng Zhang, Hai Zhao, and Dong Yu. 2024. DOCBENCH: A benchmark for evaluating LLM-based document reading systems. _arXiv preprint arXiv:2407.10701_ . 

- Jinxu Zhang, Qiyuan Fan, Yongqi Yu, and Yu Zhang. 2025. Dream: Integrating hierarchical multimodal retrieval with multi-page multimodal language model for documents vqa. In _Proceedings of the 33rd ACM International Conference on Multimedia_ , pages 4213– 4221. 

- Jinxu Zhang, Yongqi Yu, and Yu Zhang. 2024. Cream: coarse-to-fine retrieval and multi-modal efficient tuning for document vqa. In _Proceedings of the 32nd ACM International Conference on Multimedia_ , pages 925–934. 

- Qintong Zhang, Xinjie Lv, Jialong Wu, Baixuan Li, Zhengwei Tao, Guochen Yan, Huanyao Zhang, Bin Wang, Jiahao Xu, Haitao Mi, and 1 others. 2026. Docdancer: Towards agentic document-grounded information seeking. _arXiv preprint arXiv:2601.05163_ . 

13 

**==> picture [455 x 571] intentionally omitted <==**

**----- Start of picture text -----**<br>
Summary tokens Hi-VT5 (Tito et al., 2023)<br>§3.1 Page-to-Document Cross-page attention Arctic-TILT (Borchmann et al., 2025), GRAM (Blau et al., 2024)<br>Recurrent state RM-T5 (Dong et al., 2024)<br>Lightweight projector InstructDoc (Tanaka et al., 2024), LayTokenLLM (Zhu et al., 2025b)<br>§3.2 MLLM-Centric Token compression mPLUG-DocOwl2 (Hu et al., 2025), Texthawk2 (Yu et al., 2024)<br>§3 Architecture Backbone fine-tuning Docopilot (Duan et al., 2025b), DocR1 (Xiong et al., 2026)<br>Single retriever-reader M3DocRAG (Cho et al., 2024), VDocRAG (Tanaka et al., 2025)<br>§3.3 Retrieval-Augmented<br>Multi-component pipeline MDocAgent (Han et al., 2025), VisDoM (Suri et al., 2025)<br>External tool calls ViDoRAG (Wang et al., 2025a), MACT (Yu et al., 2025)<br>§3.4 Adaptive-Trajectory<br>Structured navigation SimpleDoc (Jain et al., 2025), DMap (Fu et al., 2026)<br>Volume Arctic-TILT (Borchmann et al., 2025), PDF-WuKong (Xie et al., 2024)<br>§4.1 Text Continuity RM-T5 (Dong et al., 2024), MultiDocFusion (Shin et al., 2025)<br>Heterogeneity AVIR (Li et al., 2025)<br>Resolution (compression) Texthawk2 (Yu et al., 2024), Leopard (Jia et al., 2024)<br>§4 Modalities §4.2 Visual Selective high-res DocSeeker (Yan et al., 2025), DocLens (Zhu et al., 2025a)<br>Multi-signal competition M3DocRAG (Cho et al., 2024), VDocRAG (Tanaka et al., 2025)<br>Hierarchy & reading order DocAgent (Sun et al., 2025), DMap (Fu et al., 2026)<br>§4.3 Structure Cross-page references KGP (Wang et al., 2024c), MoLoRAG (Wu et al., 2025b)<br>Spanning elements MHier-RAG (Gong et al., 2025), DFVC (Qian et al., 2026)<br>Similarity-based ColPali (Faysse et al., 2024b), SimpleDoc (Jain et al., 2025)<br>§5.1 Retrieval & Navigation<br>Relation-aware MultiDocFusion (Shin et al., 2025), MLDocRAG (Zhang and Wu, 2026)<br>CoT & structured Leopard (Jia et al., 2024), MM-R5 (Xu et al., 2026)<br>§5 Inference §5.2 Prompted Reasoning Self-reflection ViDoRAG (Wang et al., 2025a), VisDoM (Suri et al., 2025)<br>ReAct & tool-augmented Doc-V* (Zheng et al., 2026), Doc-React (Wu et al., 2025a)<br>§5.3 Agentic Methods Modality-specialised MDocAgent (Han et al., 2025), M [2] RAG (Duan et al., 2025a)<br>Role-specialised MACT (Yu et al., 2025), DocAgent (Sun et al., 2025)<br>Modality alignment VDocRAG (Tanaka et al., 2025)<br>§6.1 Retrieval Cross-page fusion DFVC (Qian et al., 2026)<br>Logical relevance MoLoRAG (Wu et al., 2025b)<br>Trace imitation DocSeeker (Yan et al., 2025), CoR (Guo et al., 2026)<br>§6.2 Reasoning Reward-driven RL DocR1 (Xiong et al., 2026), MM-R5 (Xu et al., 2026)<br>§6 Training Uncertainty calibration DocSLM (Hannan et al., 2025)<br>Trajectory distillation Doc-V* (Zheng et al., 2026), DocDancer (Zhang et al., 2026)<br>§6.3 Agentic<br>Reinforcement-learned MM-Doc-R1 (Lin et al., 2026), VRAG-RL (Wang et al., 2026)<br>PEFT LayTokenLLM (Zhu et al., 2025b), CREAM (Zhang et al., 2024)<br>§6.4 Cross-Cutting<br>Multi-stage curriculum mPLUG-DocOwl2 (Hu et al., 2025), DocSLM (Hannan et al., 2025)<br>§C.1 Pretraining DocStruct4M (Hu et al., 2024), MP-DocStruct1M (Hu et al., 2025)<br>§7 Datasets §C.2 Supervised Fine-tuning InstructDoc (Tanaka et al., 2024), DocDancer (Zhang et al., 2026)<br>§C.3 Benchmarking MMLongBench-Doc (Ma et al., 2024), LongDocURL (Deng et al., 2025)<br>§8 Efficiency & Deployability DocSLM (Hannan et al., 2025), URaG (Shi et al., 2026)<br>§8 Faithfulness & Abstention DocR1 (Xiong et al., 2026), DocLens (Zhu et al., 2025a)<br>§8 Challenges §8 Evidence Pipeline MHier-RAG (Gong et al., 2025), KGP (Wang et al., 2024c)<br>§8 Supervision & Contamination InstructDoc (Tanaka et al., 2024), DocBench (Zou et al., 2024)<br>§8 Evaluation Scope UDA (Hui et al., 2024), BRIDGE (Xiang et al., 2026)<br>Taxonomy<br>MP-VRDU<br>**----- End of picture text -----**<br>


Figure 2: MP-VRDU Taxonomy 

14 

## **A More Methodology Details** 

**Search keywords.** Queries combined a multipage cluster with a task or method cluster. Multipage terms included “multi-page document”, “long document”, “long-context document”, “cross-page reasoning”, “page-level retrieval”, “documentscale evidence”, and “hundred-page”. Task and method terms included “document visual question answering”, “visually rich document understanding”, “document VQA”, “document agent”, “visual document retrieval”, “document RAG”, “hierarchical document transformer”, and “longcontext vision-language model”. Benchmarkname probes (“MP-DocVQA”, “MMLongBenchDoc”, “LongDocURL”, “DUDE”, “SlideVQA”, “M3DocVQA”, “DocBench”, “M-LongDoc”) were used as anchors for forward-citation sweeps to capture systems that report against multi-page evaluation suites. 

**Inclusion and exclusion criteria.** A paper was retained if it (i) proposed a multi-page model, framework, system, or pipeline rather than a benchmark or dataset alone, and (ii) reported quantitative results on at least one multi-page benchmark, with cross-page evidence handling as an explicit design contribution. Excluded categories were single-page-only systems benchmarked solely on DocVQA, InfographicVQA, ChartQA, FUNSD, or CORD; survey, position, and opinion papers; generic OCR or table-extraction tools without VQA or reasoning evaluation; long-context LLM benchmarks where documents are one input type among many; and dataset-only contributions, which are catalogued separately in Appendix C. Peer-reviewed venue publications and arXiv preprints were assessed against the same substantive criteria, since recent MP-VRDU progress has appeared predominantly in preprint form; venue status is recorded per entry in Table 1. 

**Knowledge extraction and verification.** Each retained paper was converted to markdown and processed through LLM-assisted key-information extraction with evidence-span location, populating per-model fields covering architectural family, OCR dependency, vision encoder, LLM backbone, adaptors and projectors, page-rendering resolution, training stages, datasets used per stage, and reported scores on each benchmark. Every extracted field was manually verified against the source paper before being committed to the corpus tables. 

**Dataset catalogue.** The dataset catalogue in Appendix C was built primarily from the pretraining, fine-tuning, and benchmarking datasets named by every paper in the surveyed corpus, supplemented with a small set of datasets the corpus does not yet adopt but that cover MP-VRDU-relevant settings such as long scientific documents, multi-page tables, and cross-page reasoning suites, included to surface datasets whose adoption may strengthen future cross-system comparison. 

## **B More Framework Details** 

## **B.1 Multimodal Fusion** 

In multi-page VRDU, the central challenge is not only multimodal fusion itself, but determining when and how text, visual, and layout information should be selectively fused across large collections of pages under strict computational constraints. 

**Early Fusion.** Text, visual, and layout features are combined before or during document encoding. In page-to-document transformers, fusion is local: OCR tokens, bounding boxes, visual patches, and question tokens are combined within each page and then aggregated through page summaries (Tito et al., 2023), document-level tokens (Blau et al., 2024), recurrent memory (Dong et al., 2024), or sparse text-image-layout attention (Borchmann et al., 2025). Backbone-centric MLLMs apply early fusion at projection time, mapping visual or multimodal features into the LLM token space through adapters, projectors, or document-former modules (Tanaka et al., 2024; Duan et al., 2025b; Jia et al., 2024). The pattern preserves fine-grained multimodal alignment at the cost of joint computation over many pages. 

**Late Fusion.** Modalities are kept separate during retrieval or inspection and combined only after candidate evidence has been selected. Retrievergenerator systems identify relevant evidences via text chunks before visual generation (Zhang et al., 2024), perform page-image retrieval that keeps multimodal evidence fused at the page level (Cho et al., 2024), or combine parsed text, visual descriptions, parent pages, and cross-page summaries only for retrieved evidence (Gong et al., 2025). Agentic pipelines extend the pattern by inspecting text, image, table, and page-level evidence separately before final synthesis or adjudication (Han et al., 2025; Duan et al., 2025a; Zhu et al., 2025a). The pattern scales to long documents but depends on 

15 

retrieval and routing quality. 

## **B.2 Coarse-to-Fine Reasoning** 

Coarse-to-fine reasoning is the most widely adopted structured-reasoning pattern in the surveyed corpus, appearing across MLLM-centric, retrieval-augmented, and adaptive-trajectory systems (Table 1). The pattern decomposes documentscale reasoning into successive stages of decreasing scope: an initial pass over the whole document or a broad candidate set identifies regions likely to contain answer-relevant evidence, and subsequent passes restrict attention to those regions at higher fidelity. Three implementations recur. **Page to region** variants first select candidate pages from the document, then reason over each candidate at higher fidelity or with more detailed analysis before answering (Shi et al., 2026; Xiong et al., 2026; Yan et al., 2025; Gong et al., 2025), with some realisations operating across discrete passes and others embedding the pattern within the layers of a single MLLM. **Retrieve, rerank, then read** variants extend the pattern across the retrieval stack, with an initial dense or sparse retrieval returning a broad candidate set, a reranker narrowing it through LLMbased scoring, and the generator reading only the final subset (Zhang et al., 2024; Xu et al., 2026; Shin et al., 2025; Zhang and Wu, 2026). **Plan then execute** variants instantiate the pattern at the controller level, first producing an evidence-acquisition plan that identifies which pages or regions to inspect, then executing the plan in sequence with iterative refinement (Zheng et al., 2026; Sun et al., 2025; Fu et al., 2026). Across all three, early stages operate over compact representations and late stages at full fidelity over a narrowed scope, making coarse-tofine a natural fit for the cost structure of MP-VRDU, though errors at an early stage are unrecoverable at later stages since later stages see only what was admitted upstream. 

## **B.3 Training Status by Component** 

Table 5 records which components are trained, frozen, or not applicable in each surveyed system across pretraining, instruction-tuning, and fine-tuning stages. Three patterns recur. _Endto-end_ systems train multiple components jointly and dominate page-to-document encoding models. _Partially trained_ systems freeze the heavy backbone and update only a light adaptor or projection layer, the prevalent pattern across retrievergenerator and retrieval-augmented backbone vari- 

ants. _Training-free_ systems perform no parameter updates and dominate adaptive-trajectory pipelines built on prompted, multi-agent orchestration of frozen MLLMs. This view also clarifies why some scores in Table 2 are not directly comparable across the training/zero-shot boundary. 

## **B.4 Model Components Details** 

Table 6 lists the concrete components of each surveyed system, organised by architectural family. The four families assemble these parts in characteristically different ways. Page-to-document transformers pair smaller LLM backbones (T5 variants, DocFormerv2) with custom layout-aware projectors. MLLM-centric adaptation models reuse standard vision-language stacks such as Qwen2.5VL, mPLUG-Owl2, and SigLIP variants, and update only light projection or compression modules. Retriever-augmented pipelines decouple a frozen retriever (ColPali, ColQwen, VisRAG) from a frozen LVLM and train only the scorer, fuser, or reranker. Adaptive-trajectory pipelines mostly orchestrate frozen frontier VLMs, with parameter updates restricted to a few systems that fine-tune the controller via SFT or GRPO. 

## **B.5 Reported Performance by Architecture** 

Table 2 summarises reported scores on the five most-adopted multi-page benchmarks, grouped by architectural family. The matrix is sparse and metric conventions diverge across MP-DocVQA, DUDE, LongDocURL, MMLongBench-Doc, and SlideVQA, so scores are not directly comparable across columns. Coverage tracks architectural inductive bias: page-to-document transformers concentrate on mid-length benchmarks, while retrievergenerator and adaptive-trajectory systems dominate the long-document frontier where selective evidence access makes 50+ page evaluation tractable. Cross-family comparisons should also account for backbone scale, proprietary-LLM dependency, and the training-versus-zero-shot distinction made explicit in Table 5. 

## **B.6 General Domain MLLM for MP-VRDU** 

Frontier general-domain MLLMs handle short multi-page inputs well when the document fits in context, but the multi-page setting introduces three pressures absent in single-page VRDU: contextwindow overrun, attention dilution over sparsely distributed evidence (Liu et al., 2024), and a perpage reading fidelity versus whole-document cov- 

16 

|**MP-DocVQA DUDE LongDocURL MMLongBench-Doc**<br>**Model**<br>**Variant**<br>**ANLS**<br>**ANLS**<br>**Accuracy**<br>**Accuracy**<br>**F1**|**MP-DocVQA DUDE LongDocURL MMLongBench-Doc**<br>**Model**<br>**Variant**<br>**ANLS**<br>**ANLS**<br>**Accuracy**<br>**Accuracy**<br>**F1**|**MP-DocVQA DUDE LongDocURL MMLongBench-Doc**<br>**Model**<br>**Variant**<br>**ANLS**<br>**ANLS**<br>**Accuracy**<br>**Accuracy**<br>**F1**|**SlideVQA**<br>**EM**<br>**F1**|
|---|---|---|---|
|||||
|**General Domain MLLM**||||
|Claude-3-Opus<br>Claude-3-Opus<br>Gemini-1.5-Pro<br>Gemini-1.5-Pro<br>GPT-4o<br>GPT-4o (LVLM mode)<br>GPT-4V<br>GPT-4V(ision)<br>InternVL-Chat-v1.5<br>InternVL-Chat-v1.5-26B<br>LLaVA-OneVision-7B<br>LLaVA-OneVision-Chat-7B<br>MiniCPM-Llama3-V2.5-8B MiniCPM-Llama3-V2.5<br>Qwen2-VL-7B<br>Qwen2-VL-7B|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|–<br>17.4<br>18.1<br>50.9<br>28.2<br>20.6<br>64.5<br>42.8<br>44.9<br>–<br>32.4<br>31.2<br>–<br>14.6<br>13.0<br>25.0<br>–<br>–<br>–<br>8.5<br>8.6<br>30.6<br>–<br>–|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|
|**Page-to-Document Encoding Models**||||
|Arctic-TILT<br>Arctic-TILT<br>GRAM<br>GRAM-859M<br>Hi-VT5<br>Hi-VT5 (multipage)<br>RM-T5<br>RM-T5|81.2<br>58.1<br>80.32<br>51.15<br>62.01<br>–<br>64.01<br>–|–<br>25.8<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|55.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–|
|**MLLM-Centric Adaptation Models**||||
|DocR1<br>DocR1<br>DocSeeker<br>DocSeeker (full SFT + EviGRPO)<br>Leopard<br>Leopard-LLaVA-Pro<br>mPLUG-DocOwl2<br>DocOwl2-8B<br>URaG<br>URaG-7B<br>Docopilot<br>Docopilot-8B<br>Texthawk2<br>Texthawk2<br>DocSLM<br>DocSLM-2B<br>InstructDoc<br>InstructDoc (fnetuned)<br>LayTokenLLM<br>LayTokenLLM-8B<br>CoR<br>Qwen2.5-VL-CoR-7B|87.45<br>54.39<br>86.2<br>57.4<br>70.6<br>43.82<br>69.42<br>46.77<br>88.2<br>57.6<br>81.3<br>–<br>–<br>–<br>70<br>47.6<br>–<br>46.8<br>74.31<br>52<br>–<br>–|–<br>–<br>–<br>51.7<br>40.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>52.2<br>33.8<br>32.8<br>–<br>28.8<br>23<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>51.5<br>37.4<br>36|–<br>71.96<br>–<br>77.1<br>–<br>37.51<br>–<br>–<br>72.1<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>37.7<br>47.3<br>–<br>–<br>–<br>–|
|**Retrieval-Augmented Pipelines**||||
|AVIR<br>AVIR<br>DFVC<br>DFVC (ColQwen backbone)<br>DREAM<br>DREAM† (InternVL2-4B + retrieved text)<br>M3DocRAG<br>M3DOCRAG (ColPali + Qwen2-VL 7B)<br>MM-R5<br>MM-R5 (Qwen2.5-VL-7B SFT+GRPO)<br>MoLoRAG<br>MoLoRAG+ (Qwen2.5-VL-7B)<br>Self-Attention Scoring<br>Pix2Struct-SA proposed<br>SLEUTH<br>SLEUTH (Qwen3-VL-8B + ColPali Top-5)<br>SV-RAG<br>SV-RAG-InternVL2† R5 (4B)<br>VDocRAG<br>VDocRAG<br>CREAM<br>CREAM<br>MHier-RAG<br>MHier-RAG (Qwen-turbo)<br>MLDocRAG<br>MLDocRAG<br>MultiDocFusion<br>MultiDocFusion<br>M2RAG<br>M[2]RAG top-5<br>MDocAgent<br>MDocAgent (top-4)<br>PDF-WuKong<br>PDF-WuKong<br>RAG-DocVQA<br>RAG-Qwen2.5-VL<br>VisDoMRAG<br>VisDoMRAG (ChatGPT-4o + ColPali + BGE)<br>KGP<br>KGP|84.58<br>–<br>–<br>–<br>85.72<br>52.14<br>84.44<br>–<br>–<br>–<br>–<br>–<br>61.99<br>–<br>–<br>–<br>71.0<br>45.0<br>–<br>44<br>65.32<br>52.46<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>76.9<br>56.1<br>73.7<br>39.9<br>–<br>–<br>–<br>–|–<br>–<br>–<br>–<br>–<br>–<br>–<br>27.3<br>28.6<br>–<br>21<br>22.6<br>–<br>–<br>–<br>51.85<br>41.01<br>–<br>–<br>–<br>–<br>59.96<br>52.77<br>–<br>–<br>23.0<br>24.2<br><br>–<br>–<br>–<br>–<br>–<br>–<br>55.7<br>52.3<br>46<br>50.8<br>47.9<br>–<br>–<br>–<br>–<br>57.68<br>39.9<br>–<br>57.8<br>31.5<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|60.3<br>68.9<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>31.98<br>–<br>–<br>42<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>67.22<br>–<br>–|
|**Adaptive-Trajectory Pipelines**||||
|Doc-V*<br>Doc-V* (GRPO)<br>DocReact<br>DocReact (w/ ColPali)<br>MACT<br>MACT-MiMo-VL-Series-28B<br>VRAG-RL<br>VRAG-RL (Qwen2.5-VL-7B-Instruct)<br>DocLens<br>DocLens (Gemini-2.5-Pro)<br>DMAP<br>DMAP Ours-Top4 (GPT-4o backbone)<br>DocAgent<br>DocAgent (Claude-3.5-Sonnet)<br>DocDancer<br>DocDancer (GPT-5.2)<br>MM-Doc-R1<br>MM-Doc-R1 (Qwen3-8B + SPO)<br>SimpleDoc<br>SimpleDoc (3.5 pages)<br>ViDoRAG<br>ViDoRAG|86.2<br>64.5<br>–<br>–<br>–<br>70.8<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|56.3<br>42.1<br>–<br>–<br>77.2<br>–<br>38.29<br>38.07<br>54.87 55.04<br>–<br>–<br>47.4<br>–<br>83.9<br>–<br>–<br>–<br>–<br>–<br>–<br>67.6<br>–<br>–<br>–<br>60.7<br>43.2<br>–<br>–<br>–<br>–<br>57.3<br>54.1<br>–<br>–<br>–<br>57<br>56.8<br>–<br>–<br>–<br>49.7<br>46.1<br>–<br>–<br>72.3<br>60.58<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–||



Table 2: Performance of MP-VRDU models grouped by architecture category. Reported metrics follow the source papers and are not always directly comparable. ‘–’ indicates not reported. 

erage trade-off under a fixed token budget. Table 2 shows that even the strongest frontier MLLMs trail specialised MP-VRDU systems by a wide margin on MMLongBench-Doc and LongDocURL, motivating the specialised architectures examined in §3. 

tory directly. Three further systems (DocLens, RAG-DocVQA, VDocRAG) distribute through github.io project pages, MACT announces a forthcoming release without a URL, and the remaining papers do not announce open-source releases. 

## **B.7 Open-Source Code Releases** 

Table 7 lists the surveyed MP-VRDU models that release implementations through a public GitHub repository: 16 of 36 papers link such a reposi- 

## **C More Dataset Details** 

Datasets in MP-VRDU span three roles: pretraining (PT), supervised fine-tuning (SFT), and benchmarking (BM), with several datasets occupying 

17 

|**Dataset**|**Pages**|**Type**|**Venue**|**Domain**|**# Docs**|**Avg Pages**|**# Q/A**|**Multi/S Hop**|
|---|---|---|---|---|---|---|---|---|
|DocVQA (2021)|SP|PT+BM|WACV|Industry|12767|–|50,000|S|
|FeTaQA (2022)|SP|PT+BM|TACL|Wikipedia tables|10330|–|10,330|M|
|InfographicVQA (2022)|SP|PT+BM|WACV|Infographics|5,485|–|30,035|M+S|
|PDF-VQA Task A (2023)|SP|PT+BM|ECML-PKDD|Biomedical|–|–|81,085|S|
|PDF-VQA Task B (2023)|SP|PT+BM|ECML-PKDD|Biomedical|–|–|53,872|S|
|TabFact (2020)|SP|PT+BM|ICLR|Wikipedia tables|16573|–|118,275|M+S|
|TextVQA (2019)|SP|PT+BM|CVPR|Scene-text images|28408|–|45,336|S|
|VisualMRC (2021)|SP|PT+BM|AAAI|Webpages|10197|–|30,562|M+S|
|WikiTableQuestions (2015)|SP|PT+BM|ACL|Wikipedia tables|2108|–|22,033|M|
|ChartQA (2022)|SP|BM|ACL|Charts|20882|–|32,719|M+S|
|DocGenome (2024)|MP|PT+BM|NeurIPS|Academic Paper|500000|13|3000|M+S|
|DuReaderVis (2022)|MP|PT+BM|ACL|Cross-domain|158000|–|15,000|S|
|Kleister-Charity (2021)|MP|PT+BM|ICDAR|Financial|2778|22.19|21,612|S|
|Kleister-NDA (2021)|MP|PT+BM|ICDAR|Legal|540|5.98|2,160|S|
|M-LongDoc (2025)|MP|PT+BM|EMNLP|Cross-domain|180|210.8|851|S|
|MMVQA (2024)|MP|PT+BM|IJCAI|Biomedical|3146|9.6|262,928|M+S|
|MP-DocVQA (2023)|MP|PT+BM|PR|Industry|5928|8.27|46,176|S|
|MultiHiertt (2022)|MP|PT+BM|ACL|Financial reports|2513|~2.5|10,440|M|
|PaperPDF (2024)|MP|PT+BM|preprint|Academic Paper|–|–|1,110,000|M+S|
|PDF-VQA Task C (2023)|MP|PT+BM|ECML-PKDD|Biomedical|1147|~10|5,653|M|
|QASPER (2021)|MP|PT+BM|NAACL|Academic Paper|1585|~8|5,049|M|
|SlideVQA (2023)|MP|PT+BM|AAAI|Presentations|2619|20|14,484|M+S|
|MP-DocStruct1M (2025)|MP|PT|ACL|Cross-domain|–|–|1,113,259|S|
|OpenDocVQA (2025)|MP|SFT+BM|CVPR|Cross-domain|–|–|–|M+S|
|CoR-Dataset (2026)|MP|SFT|preprint|Cross-domain|–|–|26,088|M+S|
|Doc-750K (2025b)|MP|SFT|CVPR|Academic Paper|–|–|758,000|M+S|
|MHDocVQA (2025)|MP|SFT|CVPR|Cross-domain|–|9.5|38,020|M|
|MP-DocReason51K (2025)|MP|SFT|ACL|Cross-domain|–|–|51,754|M+S|
|BRIDGE (2026)|MP|BM|preprint|Academic Paper|262|–|11,857|M|
|DocBench (2024)|MP|BM|preprint|Cross-domain|229|–|1,102|S|
|DUDE (2023)|MP|BM|ICCV|Cross-domain|5019|5.72|41,541|M+S|
|FetaTab (2024)|MP|BM|NeurIPS|Wikipedia Articles|871|~11|1,016|S|
|LongDocURL (2025)|MP|BM|ACL|Cross-domain|396|85.6|2,325|M+S|
|M3DocVQA (2025)|MP|BM|CVPR|Wikipedia Articles|3368|12.2|2,441|M+S|
|MM-NIAH (2024b)|MP|BM|NeurIPS|Synthetic|–|–|16,486|M+S|
|MMDocIR (2025)|MP|BM|ACL|Cross-domain|313|65.1|1,658|M+S|
|MMLongBench (2025b)|MP|BM|NeurIPS|Cross-domain|–|–|13,331|M+S|
|MMLongBench-Doc (2024)|MP|BM|NeurIPS|Cross-domain|135|47.5|1,082|M+S|
|PaperTab (2024)|MP|BM|NeurIPS|Academic Paper|307|~11|393|M+S|
|PaperText (2024)|MP|BM|NeurIPS|Academic Paper|1087|~11|2,804|M+S|
|ViDoSeek (2025a)|MP|BM|EMNLP|Cross-domain|–|–|1,200|M+S|
|VisDoMBench (2025)|MP|BM|NAACL|Cross-domain|1277|19.11|2,271|M+S|
|VisR-Bench (2025)|MP|BM|ICLR|Cross-domain|226|–|471|M+S|
|DocStruct4M (2024)|MP+SP|PT+SFT|EMNLP|Cross-domain|–|–|4,000,000|M+S|
|TAT-DQA (2022)|MP+SP|PT+BM|ACM|Financial reports|2758|1.33|16,558|M|
|InstructDoc (2024)|MP+SP|SFT|AAAI|Cross-domain|–|–|–|M+S|
|Leopard-Instruct (2024)|MP+SP|SFT|TMLR|Cross-domain|–|–|925,000|M+S|



Table 3: Datasets referenced in the MP-VRDU survey. **Pages** : document scope (MP = Multi-Page, SP = Single-Page, MP+SP = Mixed). **Type** : dataset role – PT (Pretraining), SFT (Supervised Fine-Tuning), BM (Benchmark), and combinations. **Hop** : question-hop type (Single, Multi, or both). ‘–’ indicates not reported. 

more than one role across different systems. Perdataset statistics are available in Table 3 and perpaper adoption in Table 8. 

## **C.1 Pretraining** 

Pretraining is the exception rather than the norm in MP-VRDU. Only a handful of page-to-document and MLLM-centric systems pretrain or continuetrain a backbone at all, while retrieval-augmented and agentic pipelines inherit pretrained visionlanguage backbones and skip the stage entirely. Two factors explain its scarcity. Document-scale pretraining is computationally expensive, and genuinely multi-page pretraining corpora are few, so most systems that do pretrain rely on single-page archives or repurpose single-page data. Early work used single-page OCR archives such as IITCDIP (Lewis et al., 2006) and OCR-IDL (Biten 

et al., 2022), which supply text-recognition supervision but no cross-page structure. The small number of genuinely multi-page corpora appeared only recently, as DocStruct4M (Hu et al., 2024) mixes single- and multi-page samples with structureaware parsing supervision, and the purely multipage MP-DocStruct1M (Hu et al., 2025), Doc750K (Duan et al., 2025b), and bilingual PaperPDF corpora (Xie et al., 2024) support multi-image continued pretraining over genuine document inputs. Where multi-page pretraining does occur, the objective has moved from text recognition toward structure-aware supervision, though the corpus as a whole shows that most cross-page capability is instilled at fine-tuning rather than pretraining. 

18 

## **C.2 Supervised Fine-tuning** 

Per-page human annotation does not scale to the page counts typical of MP-VRDU, so supervised fine-tuning has converged on two strategies that are often combined within a single system. The first strategy aggregates publicly released questionanswer collections into broad instruction mixes that exercise heterogeneous question and answer types (Tanaka et al., 2024; Jia et al., 2024; Hu et al., 2025; Duan et al., 2025b). The second strategy curates a bespoke corpus for the system under study, typically synthesised by a strong visionlanguage model writing question-answer pairs, reasoning traces, or agent trajectories over scraped PDFs (Xie et al., 2024; Tanaka et al., 2025; Wu et al., 2025b; Zhang et al., 2026; Guo et al., 2026). Synthetic generation has become the dominant scaling path for fine-tuning data, and the underlying PDF pools are typically drawn from the same scientific archives, slide repositories, and benchmark document sources that also supply evaluation. Layered above supervised fine-tuning, a growing body of work applies reinforcement or preference learning with shaped rewards that credit correct evidence-page citation, faithful chain-of-thought structure, or appropriate tool use rather than finalanswer accuracy alone (Xiong et al., 2026; Guo et al., 2026; Yan et al., 2025; Lin et al., 2026; Xu et al., 2026; Wang et al., 2026). The supervision pools at this stage remain small relative to the supervised mixes that precede them, indicating that reinforcement signals are used for behaviour shaping; instilling evidence grounding, output formatting, and multi-step reasoning patterns, rather than for further scaling. 

## **C.3 Benchmarking** 

Benchmarks have evolved through several tiers that have accumulated rather than replaced one another. Single-page extractive evaluation on DocVQA (Mathew et al., 2021), InfographicVQA (Mathew et al., 2022), and ChartQA (Masry et al., 2022) persists as a per-page baseline, with mid-length multi-page corpora such as MPDocVQA (Tito et al., 2023), DUDE (Van Landeghem et al., 2023), and SlideVQA (Tanaka et al., 2023) extending the same extractive answer formats into a few-page setting. Long-document multimodal benchmarks then introduced documents whose total length routinely exceeds backbone context windows and whose evidence requires cross- 

|**Dataset**|**PT**|**SFT**|**BM**|**SFT**_∩_**BM**|
|---|---|---|---|---|
|MP-DocVQA|0|19|20|16|
|DUDE|0|16|20|12|
|MMLongBench-Doc|0|3|25|2|
|DocVQA|0|10|14|8|
|SlideVQA|0|7|17|6|
|InfographicVQA|0|9|12|7|
|LongDocURL|0|3|14|2|
|ChartQA|0|5|7|4|
|VisualMRC|0|4|5|2|
|FetaTab|0|0|6|0|
|PaperTab|0|0|6|0|
|TabFact|0|2|3|1|



Table 4: Top 12 datasets by total appearance count across the surveyed corpus, with usage counts broken down by role. **SFT** _∩_ **BM** counts the papers that use the dataset in _both_ fine-tuning and benchmarking. 

page integration over mixed text, tables, charts, and figures, including MMLongBench-Doc (Ma et al., 2024), LongDocURL (Deng et al., 2025), DocBench (Zou et al., 2024), MMDocIR (Dong et al., 2025), and M-LongDoc (Chia et al., 2025), while retrieval-oriented benchmarks such as UDABenchmark (Hui et al., 2024), DocR1 (Xiong et al., 2026), and DocLens (Zhu et al., 2025a) additionally score evidence-page selection and layout-level localisation alongside answer accuracy. The most recent additions push further into cross-document settings such as M3DocRAG (Cho et al., 2024), ViDoRAG (Wang et al., 2025a), VDocRAG (Tanaka et al., 2025), and VisDoM (Suri et al., 2025), while also expanding beyond the academic, financial, and slide-deck domains that dominated earlier work through datasets such as SVRAG (Chen et al., 2025). Despite this proliferation, evaluation in practice has consolidated around a small set of long-document and mid-length benchmarks that recur in most reported comparisons and serve as the de-facto standards for cross-model assessment. 

## **C.4 Contamination Matrix** 

Table 4 quantifies the contamination risk discussed in §8. Overlap between fine-tuning and benchmarking is highest on mid-length corpora: 16 of the 19 papers that fine-tune on MP-DocVQA also evaluate on it, with comparable patterns for DUDE, SlideVQA, and InfographicVQA. Long-document benchmarks show smaller overlap so far, though their fine-tuning adoption is rising. 

19 

|**Model**<br>**Arch.**|**Vision Encoder**<br>**PT**<br>**IT**<br>**FT**|**LLM Backbone**<br>**PT**<br>**IT**<br>**FT**|**Adaptors**<br>**PT**<br>**IT**<br>**FT**|
|---|---|---|---|
|**End-to-End**||||
|Arctic-TILT<br>PDE<br>GRAM<br>PDE<br>Docopilot<br>MLLM<br>DocR1<br>MLLM<br>DocSeeker<br>MLLM<br>DocSLM<br>MLLM<br>mPLUG-DocOwl2<br>MLLM<br>Texthawk2<br>MLLM<br>PDF-WuKong<br>RA<br>Doc-V*<br>AT<br>DocDancer<br>AT<br>MACT<br>AT<br>VRAG-RL<br>AT|✓<br>–<br>✓<br>–<br>–<br>✓<br>–<br>✓<br>–<br>–<br>–<br>✓<br>–<br>✓<br>✓<br>✓<br>_×_<br>_×_<br>✓<br>–<br>_×_<br>✓<br>✓<br>–<br>–<br>–<br>✓<br>–<br>_×_<br>_×_<br>–<br>_×_<br>–<br>–<br>✓<br>✓<br>–<br>✓<br>✓|✓<br>–<br>✓<br>–<br>–<br>✓<br>–<br>✓<br>–<br>–<br>–<br>✓<br>–<br>✓<br>✓<br>_×_<br>✓<br>✓<br>_×_<br>–<br>✓<br>✓<br>✓<br>–<br>–<br>–<br>✓<br>–<br>✓<br>✓<br>–<br>✓<br>–<br>–<br>✓<br>✓<br>–<br>✓<br>✓|✓<br>–<br>✓<br>–<br>–<br>✓<br>–<br>✓<br>–<br>–<br>–<br>–<br>–<br>✓<br>✓<br>✓<br>✓<br>✓<br>✓<br>–<br>✓<br>✓<br>✓<br>–<br>–<br>–<br>✓<br>–<br>_×_<br>_×_<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>✓<br>✓|
|**Partially Trained**||||
|Hi-VT5<br>PDE<br>RM-T5<br>PDE<br>CoR<br>MLLM<br>InstructDoc<br>MLLM<br>LayTokenLLM<br>MLLM<br>Leopard<br>MLLM<br>URaG<br>MLLM<br>AVIR<br>RA<br>CREAM<br>RA<br>DFVC<br>RA<br>DREAM<br>RA<br>KGP<br>RA<br>M2RAG<br>RA<br>MLDocRAG<br>RA<br>MM-R5<br>RA<br>MoLoRAG<br>RA<br>MultiDocFusion<br>RA<br>RAG-DocVQA<br>RA<br>Self-Attention Scoring<br>RA<br>SV-RAG<br>RA<br>VDocRAG<br>RA<br>MM-Doc-R1<br>AT|_×_<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>_×_<br>_×_<br>–<br>_×_<br>_×_<br>–<br>–<br>–<br>_×_<br>_×_<br>–<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>–<br>–<br>–<br>_×_<br>–<br>–<br>✓<br>–<br>_×_<br>_×_<br>–<br>–<br>✓<br>–<br>–<br>–<br>–<br>–<br>✓<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>_×_<br>–<br>_×_<br>–<br>–<br>_×_|✓<br>–<br>✓<br>✓<br>–<br>✓<br>–<br>_×_<br>✓<br>–<br>_×_<br>_×_<br>_×_<br>_×_<br>–<br>_×_<br>✓<br>–<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>✓<br>✓<br>–<br>–<br>–<br>_×_<br>–<br>–<br>✓<br>–<br>✓<br>✓<br>–<br>–<br>✓<br>–<br>✓<br>–<br>–<br>–<br>✓<br>–<br>–<br>_×_<br>–<br>–<br>_×_<br>_×_<br>–<br>_×_<br>–<br>–<br>✓|✓<br>–<br>✓<br>✓<br>–<br>✓<br>–<br>✓<br>✓<br>–<br>✓<br>✓<br>✓<br>✓<br>–<br>✓<br>✓<br>–<br>–<br>–<br>✓<br>–<br>–<br>✓<br>–<br>–<br>✓<br>–<br>–<br>✓<br>–<br>–<br>✓<br>–<br>✓<br>–<br>–<br>–<br>✓<br>–<br>–<br>–<br>–<br>✓<br>✓<br>–<br>–<br>–<br>–<br>✓<br>–<br>–<br>–<br>✓<br>–<br>–<br>✓<br>–<br>–<br>✓<br>✓<br>–<br>✓<br>–<br>–<br>✓|
|**Training-Free**||||
|M3DocRAG<br>RA<br>MDocAgent<br>RA<br>MHier-RAG<br>RA<br>SLEUTH<br>RA<br>VisDoMRAG<br>RA<br>DMAP<br>AT<br>DocAgent<br>AT<br>DocLens<br>AT<br>DocReact<br>AT<br>SimpleDoc<br>AT<br>ViDoRAG<br>AT|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–<br>–|



Table 5: Training status of MP-VRDU model components grouped by training category. ✓denotes trained parameters, _×_ denotes frozen parameters, and ‘–’ indicates not reported / not applicable. Architectural family abbreviations: PDE = Page-to-Document Encoding Models; MLLM = MLLM-Centric Adaptation Models; RA = Retrieval-Augmented Pipelines; AT = Adaptive-Trajectory Pipelines. 

20 

Table 6: Per-model components of the surveyed MP-VRDU systems. Resolution lists the input image resolution or pagerendering setting; _dynamic_ denotes encoders that accept variable-resolution input. ‘–’ indicates not reported not applicable. 

|**Model**|**Vision Backbone**|**Resolution**|**LLM Backbone**|**Adaptors and Projectors**|
|---|---|---|---|---|
|**Page-to-Document Encoding Models**|||||
|Arctic-TILT|U-Net|400k tokens / 500p|T5-Large|TILT fusion|
|GRAM|DocFormerv2|800 tok/page train, 8000 test|DocFormerv2|–|
|Hi-VT5|DiT|224x224 (DiT default), 20480|T5|–|
|||max tokens|||
|RM-T5|DiT|512 tok/page max|T5-base|–|
|**MLLM-Centric Adaptation Models**|||||
|DocR1|Qwen2.5-VL|dynamic, max 1024 patches|Qwen2.5-VL-7B|Frozen Qwen MLP projector|
|DocSeeker|Qwen2.5-VL ViT|1024x784 eval, EGRA training|Qwen2.5-VL-7B-Instruct|–|
|Leopard|SigLIP-SO400M|364x364 / 980x980, max 50|LLaMa-3.1-8B / Mistral-7B|MLP / Idefcs2 resampler|
|||sub-imgs|||
|mPLUG-DocOwl2|ViT-L/14|504x504, max 12 crops|mPLUG-Owl2|H-Reducer + DocCompressor|
|URaG|Qwen2.5-VL ViT|dynamic|Qwen2.5-VL-3B / Qwen2.5-VL-7B|Cross-modal retrieval module (2 linear +|
|||||GELU) + LoRA|
|Docopilot|InternViT-300M|448 tile, max 24 tiles|InternLM2-1.8B / InternLM2.5-7B|2-layer MLP|
|Texthawk2|SigLIP-SO400M|224x224 base, max 72 sub-imgs|Qwen2-7B-Instruct|QPN + ReSA|
|DocSLM|SigLIP2|dynamic 384-1152, 576 tok/page|Qwen2.5-1.5B|OCR/Vision compressor + MLP|
|InstructDoc|CLIP|224x224|FlanT5 / BLIP2|Doc-former + proj|
|LayTokenLLM|–|–|LLaMA3-8B / Qwen1.5-7B|Layout tokenizer + LoRA|
|CoR|Qwen2.5-VL|dynamic, 32k max|Qwen2.5-VL-7B|Frozen Qwen MLP projector + LoRA|
|**Retrieval-Augmented**|**Pipelines**||||
|AVIR|Pix2Struct|dynamic, 768-5120 patches|Qwen2.5-VL-3B-AWQ|Frozen Qwen MLP projector|
|DFVC|VisRAG, ColQwen|dynamic|Qwen2-VL-2B-Instruct|Cross-page MLP|
|DREAM|InternVL2 ViT|448 tile, max 24 tiles|InternVL2-4B|Cross-page MoE attention connector +|
|||||LoRA|
|M3DocRAG|ColPali / ColQwen|144 DPI, ~1700px wide|Qwen2-VL-7B / InternVL2-8B /|–|
||||Idefcs2-8B / Idefcs3-8B||
|MM-R5|Qwen2.5-VL ViT|dynamic|Qwen2.5-VL-7B|–|
|MoLoRAG|ColPali|448x448|Qwen2.5-VL 3B/7B /|Frozen base projector|
||||Deepseek-VL-16B / LlaVa-Next-7B||
|Self-Attention Scoring|Pix2Struct|2048 patches|Pix2Struct|–|
|SLEUTH|ColPali, Qwen3-VL ViT|448x448|Qwen3-VL-8B /|–|
||||GLM-4.1V-Thinking-8B /||
||||Gemini-2.5-Flash||
|SV-RAG|InternVL2 / PaliGemma|dynamic (InternVL2 default),|InternVL2-4B / PaliGemma-3B /|Dual LoRA adapters (retrieval + QA) +|
||/ Phi-3-vision|448x448|Phi-3-vision-4B|Col-projection|
|VDocRAG|Phi3V-4.2B|336x336 dynamic, up to|Phi3V-4.2B|LoRA + MLP|
|||1344x1344|||
|CREAM|Pix2Struct|2048 patches|LLaMA2-7B|Attn pooling + MLP + LoRA|
|KGP|–|–|T5-large, LLaMa-7B, MDR|LoRA|
||||(RoBERTa), ChatGPT||
|MHier-RAG|Qwen-VL-Plus|dynamic|Qwen-VL-Plus / GPT-3-turbo /|–|
||||Qwen-turbo / GPT-4o /||
||||DeepSeek-chat / ERNIE-turbo||
|MLDocRAG|LVLM Qwen2.5-VL|dynamic|Qwen2.5-VL-32B-Instruct,|Frozen Qwen MLP projector|
||||Qwen2.5-VL-7B, Qwen2.5-72B||
|MultiDocFusion|–|–|Mistral-8B / Llama-3.2-3B /|LoRA|
||||Qwen-2.5-3B / Qwen-2.5-7B||
|||||_Continued on next page_|



21 

_Table 6 – continued from previous page_ 

|**Model**|**Vision Backbone**|**Resolution**|**LLM Backbone**|**Adaptors and Projectors**|
|---|---|---|---|---|
|M2RAG|VisRAG-Ret|dynamic|Qwen2.5-7B-Instruct,|Frozen Qwen MLP projector + LoRA +|
||||Qwen2.5-VL-7B-Instruct,|modal fuser|
||||Qwen2.5-VL-32B||
|MDocAgent|ColPali / ColQwen2|144 DPI|LLaMA-3.1-8B-Instruct,|–|
||||Qwen2-VL-7B-Instruct||
|PDF-WuKong|IXC2-VL-4KHD|dynamic 336px-4K, max 16 tiles|IXC2-VL-4KHD|Sparse sampler|
|RAG-DocVQA|DiT / Pix2Struct|2048 patches|VT5 / Qwen2.5-VL-7B-Instruct|LoRA|
|VisDoMRAG|ColPali / ColQwen2|448x448 (ColPali), dynamic|Qwen2-VL-7B-Instruct /|–|
||||Gemini-1.5-Flash / GPT-4o||
|**Adaptive-Trajectory**|**Pipelines**||||
|Doc-V*|Qwen2.5-VL ViT|1024x768 / 256x256 thumb|Qwen2.5-VL-7B-Instruct|Frozen MLP|
|DocReact|ColPali, VisRAG-Ret|448x448 (ColPali default) /|GPT-4o|–|
|||dynamic|||
|MACT|Qwen2.5-VL-7B-|dynamic|Qwen2.5-VL-7B /|–|
||Instruct /||MiMo-VL-7B-SFT / InternVL3-9B,||
||MiMo-VL-7B-SFT /||Qwen2.5-7B/3B / MiMo-7B-SFT /||
||InternVL3-9B||InternVL3-8B/2B||
|VRAG-RL|Qwen2.5-VL ViT|dynamic|Qwen2.5-VL-3B-Instruct /|–|
||||Qwen2.5-VL-7B-Instruct||
|DocLens|frontier VLMs|dynamic|Gemini-2.5-Pro / Gemini-2.5-Flash|–|
||||/ Claude-4-Sonnet||
|DMAP|ColPali|dynamic|GPT-4o / Qwen-plus|–|
|DocAgent|–|144 DPI|Gemini 2.0 Flash / GPT-4o / Claude|–|
||||Sonnet 3.5||
|DocDancer|Qwen3-VL-235B-|144 DPI|Qwen3-4B/30B-A3B|–|
||A22B-Instruct||||
|MM-Doc-R1|Qwen2.5-VL ViT|dynamic|Qwen3-4B / Qwen3-8B|–|
|SimpleDoc|ColQwen-2.5|dynamic|Qwen2.5-VL-32B-Instruct,|–|
||||Qwen3-30B-A3B||
|ViDoRAG|Nv-embed-V2,|dynamic|Qwen2.5-VL / LLaMA3.2-Vision /|–|
||ColQwen2||GPT-4o||



22 

|**Model**|**GitHub Repository**|
|---|---|
|Arctic-TILT2025|https://github.com/Snowflake-Labs/arctic-tilt|
|AVIR2025|https://github.com/Li-yachuan/AVIR-main|
|DMAP2026|https://github.com/Forlorin/DMAP|
|DocAgent2025|https://github.com/lisun-ai/DocAgent|
|Docopilot2025b|https://github.com/OpenGVLab/Docopilot|
|DocSeeker2025|https://github.com/yh-hust/DocSeeker|
|DocSLM2025|https://github.com/Tanveer81/DocSLM|
|Hi-VT52023|https://github.com/rubenpt91/MP-DocVQA-Framework|
|InstructDoc2024|https://github.com/nttmdlab-nlp/InstructDoc|
|KGP2024c|https://github.com/YuWVandy/KG-LLM-MDQA|
|Leopard2024|https://github.com/tencent-ailab/Leopard|
|MDocAgent2025|https://github.com/aiming-lab/MDocAgent|
|MM-R52026|https://github.com/i2vec/MM-R5|
|MoLoRAG2025b|https://github.com/WxxShirley/MoLoRAG|
|mPLUG-DocOwl22025|https://github.com/X-PLUG/mPLUG-DocOwl|
|PDF-WuKong2024|https://github.com/yhhust/PDF-Wukong|
|Self-Attention Scoring2024|https://github.com/leitro/SelfAttnScoring-MPDocVQA|
|SimpleDoc2025|https://github.com/ag2ai/SimpleDoc|
|URaG2026|https://github.com/shi-yx/URaG|
|ViDoRAG2025a|https://github.com/Alibaba-NLP/ViDoRAG|
|VisDoMRAG2025|https://github.com/MananSuri27/VisDoM|
|VRAG-RL2026|https://github.com/Alibaba-NLP/VRAG|



Table 7: Open-source GitHub repositories released for surveyed MP-VRDU models. 

Table 8: Per-dataset usage across the surveyed corpus, ‘-’ indicates no recorded usage. Only representative datasets from Table 3 used by at least 1 paper are included, pretraining datasets are deliberately excluded. 

|**Dataset**|**Fine-Tuning**|**Benchmarking**|
|---|---|---|
|ChartQA|Docopilot, DocR1, mPLUG-DocOwl2, PDF-WuKong,|CoR, Docopilot, DocR1, MACT, PDF-WuKong, Texthawk2,|
||VDocRAG|VDocRAG|
|CoR-Dataset|CoR|-|
|Doc-750K|Docopilot|-|
|DocBench|-|DocAgent, DocDancer, M2RAG|
|DocGenome|mPLUG-DocOwl2|-|
|DocStruct4M|mPLUG-DocOwl2|-|
|DocVQA|Arctic-TILT, CREAM, Docopilot, DocR1, LayTokenLLM,|Arctic-TILT, CoR, CREAM, DocR1, DocReact, Hi-VT5,|
||mPLUG-DocOwl2, PDF-WuKong, Self-Attention Scoring,|LayTokenLLM, Leopard, MACT, mPLUG-DocOwl2,|
||SV-RAG, VDocRAG|PDF-WuKong, Self-Attention Scoring, SV-RAG, Texthawk2|
|DUDE|Arctic-TILT, CREAM, Doc-V*, DocDancer, Docopilot,|Arctic-TILT, AVIR, CREAM, Doc-V*, DocR1, DocSeeker,|
||DocR1, DocSeeker, DREAM, GRAM, LayTokenLLM,|DocSLM, DREAM, GRAM, LayTokenLLM, Leopard,|
||MM-R5, mPLUG-DocOwl2, PDF-WuKong, RAG-DocVQA,|M3DocRAG, MACT, MLDocRAG, MultiDocFusion,|
||URaG, VDocRAG|PDF-WuKong, RAG-DocVQA, SV-RAG, URaG, VDocRAG|
|FetaTab|-|DMAP, MDocAgent, MoLoRAG, SimpleDoc, SLEUTH,|
|||VisDoMRAG|



_Continued on next page_ 

23 

_Table 8 – continued from previous page_ 

|**Dataset**|**Fine-Tuning**|**Benchmarking**|
|---|---|---|
|InfographicVQA|Arctic-TILT, CREAM, Docopilot, DocR1,|Arctic-TILT, CREAM, Docopilot, DocR1, Hi-VT5,|
||mPLUG-DocOwl2, PDF-WuKong, RAG-DocVQA, SV-RAG,|InstructDoc, M3DocRAG, MACT, PDF-WuKong,|
||VDocRAG|RAG-DocVQA, Texthawk2, VDocRAG|
|InstructDoc|InstructDoc|InstructDoc|
|Kleister-Charity|Arctic-TILT, DocR1, mPLUG-DocOwl2|Arctic-TILT|
|Kleister-NDA|Arctic-TILT|Arctic-TILT|
|Leopard-Instruct|Leopard|-|
|LongDocURL|DocDancer, MM-Doc-R1, MoLoRAG|CoR, DMAP, Doc-V*, DocDancer, DocLens, DocSeeker,|
|||M2RAG, MDocAgent, MHier-RAG, MLDocRAG,|
|||MoLoRAG, SimpleDoc, SLEUTH, URaG|
|M3DocVQA|-|M3DocRAG|
|MHDocVQA|VDocRAG|-|
|MM-NIAH|-|Docopilot|
|MMDocIR|-|MM-R5|
|MMLongBench-Doc|DFVC, MoLoRAG, VRAG-RL|Arctic-TILT, CoR, DMAP, Doc-V*, DocAgent, DocDancer,|
|||DocLens, Docopilot, DocReact, DocSeeker, DocSLM,|
|||DREAM, M2RAG, M3DocRAG, MACT, MDocAgent,|
|||MHier-RAG, MLDocRAG, MM-Doc-R1, MoLoRAG,|
|||SimpleDoc, SLEUTH, SV-RAG, URaG, VRAG-RL|
|MP-DocReason51K|mPLUG-DocOwl2|-|
|MP-DocVQA|CREAM, DFVC, Doc-V*, Docopilot, DocR1, DocSeeker,|Arctic-TILT, AVIR, CREAM, DFVC, Doc-V*, DocR1,|
||DocSLM, DREAM, GRAM, Hi-VT5, LayTokenLLM,|DocSeeker, DocSLM, DREAM, GRAM, Hi-VT5,|
||M2RAG, MM-R5, mPLUG-DocOwl2, PDF-WuKong,|LayTokenLLM, Leopard, M3DocRAG, mPLUG-DocOwl2,|
||RAG-DocVQA, RM-T5, Self-Attention Scoring, URaG|PDF-WuKong, RAG-DocVQA, RM-T5, Self-Attention|
|||Scoring, URaG|
|MultiHiertt|DocR1, Leopard|DocR1, Leopard|
|OpenDocVQA|-|VDocRAG|
|PaperPDF|PDF-WuKong|PDF-WuKong|
|PaperTab|-|DMAP, MDocAgent, MoLoRAG, SimpleDoc, SLEUTH,|
|||VisDoMRAG|
|PaperText|-|DMAP, MDocAgent, SLEUTH|
|SlideVQA|DocR1, MM-R5, SV-RAG, URaG, VDocRAG, ViDoRAG,|Arctic-TILT, AVIR, Doc-V*, DocR1, DocReact, DocSeeker,|
||VRAG-RL|Leopard, M2RAG, MACT, MDocAgent, MHier-RAG,|
|||MLDocRAG, SV-RAG, URaG, VDocRAG, ViDoRAG,|
|||VRAG-RL|
|TabFact|DocR1, mPLUG-DocOwl2|DocR1, InstructDoc, Texthawk2|
|TAT-DQA|Arctic-TILT, MM-R5, SV-RAG|Arctic-TILT|
|TextVQA|DocR1, mPLUG-DocOwl2|DocR1, Leopard, Texthawk2|
|ViDoSeek|VRAG-RL|ViDoRAG, VRAG-RL|
|VisDoMBench|-|VisDoMRAG|
|VisR-Bench|-|SV-RAG|
|VisualMRC|CREAM, DocR1, mPLUG-DocOwl2, VDocRAG|CREAM, DocR1, Hi-VT5, InstructDoc, MACT|
|WikiTableQuestions|DocR1, mPLUG-DocOwl2|DocR1, Texthawk2|



24 

