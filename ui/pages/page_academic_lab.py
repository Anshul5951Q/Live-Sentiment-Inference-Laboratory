"""Page 9: Academic Learning Lab (Units U1 - U5 Syllabus Alignment)."""
import streamlit as st
from ui.components import render_header


def render():
    render_header(
        title="Academic Learning Lab & Theoretical Syllabus (U1 - U5)",
        subtitle="Mathematical formulations, conceptual paradigms, and syllabus alignment for Generative vs Discriminative NLP",
        unit_tags=["P4 Mini-Project", "Units U1-U5", "Mathematical Foundations", "KaTeX Equations", "Curriculum Traceability"],
    )

    tab_u1, tab_u2, tab_u3, tab_u4, tab_u5 = st.tabs([
        "📚 U1: ML & Features",
        "🧠 U2: Deep Sequential (RNN/LSTM)",
        "⚡ U3: Transformers & BERT",
        "🔮 U4: Generative LLMs",
        "⚖️ U5: Responsible AI & Calibration",
    ])

    # UNIT 1
    with tab_u1:
        st.markdown("### 📘 Unit 1: Foundations of Machine Learning & Text Representations")
        st.markdown(
            r"""
            #### 1. Term Frequency - Inverse Document Frequency (TF-IDF)
            In linear discriminative NLP, unstructured text is projected into a sparse, high-dimensional vector space $\mathbb{R}^{|V|}$.
            The scikit-learn smooth TF-IDF weighting formulation implemented in SentimentLab AI is:

            $$\text{tf-idf}(t, d, D) = \text{tf}(t, d) \times \left( \ln \left( \frac{1 + |D|}{1 + \text{df}(D, t)} \right) + 1 \right)$$

            Each document vector is then $L_2$-normalized:

            $$v_{\text{norm}} = \frac{v}{\|v\|_2} = \frac{v}{\sqrt{\sum_{i=1}^{|V|} v_i^2}}$$

            #### 2. Multiclass Logistic Regression (Multinomial Cross-Entropy)
            For $K=3$ discrete classes $\mathcal{Y} = \{0, 1, 2\}$, the model defines linear score functions $z_k = w_k^T x + b_k$ 
            and computes posterior class probabilities via the Softmax transformation:

            $$P(Y = c \mid X = x) = \frac{\exp(w_c^T x + b_c)}{\sum_{k=0}^{K-1} \exp(w_k^T x + b_k)}$$

            The model parameters are optimized by minimizing the categorical cross-entropy loss with $L_2$ regularization:

            $$\mathcal{L}(W) = - \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} y_{ik} \ln(p_{ik}) + \frac{1}{2C} \sum_{k=0}^{K-1} \|w_k\|_2^2$$
            """
        )

    # UNIT 2
    with tab_u2:
        st.markdown("### 🧠 Unit 2: Deep Sequential Representations & Bidirectional LSTM")
        st.markdown(
            r"""
            #### 1. The Bottleneck of Standard RNNs & The LSTM Solution
            Standard Recurrent Neural Networks suffer from exponential vanishing and exploding gradients when backpropagating through time:

            $$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=t+1}^T W_{hh}^T \text{diag}(1 - \tanh^2(a_k))$$

            Long Short-Term Memory (LSTM) introduces an additive cell state $c_t$ governed by input, forget, and output gating mechanisms:

            $$f_t = \sigma(W_f x_t + U_f h_{t-1} + b_f) \quad \text{(Forget Gate)}$$
            $$i_t = \sigma(W_i x_t + U_i h_{t-1} + b_i) \quad \text{(Input Gate)}$$
            $$\tilde{c}_t = \tanh(W_c x_t + U_c h_{t-1} + b_c) \quad \text{(Candidate Memory)}$$
            $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t \quad \text{(Cell State)}$$
            $$o_t = \sigma(W_o x_t + U_o h_{t-1} + b_o) \quad \text{(Output Gate)}$$
            $$h_t = o_t \odot \tanh(c_t) \quad \text{(Hidden State)}$$

            #### 2. Bidirectional Recurrence
            Bidirectional LSTMs process the token sequence in both forward ($\overrightarrow{h_t}$) and backward ($\overleftarrow{h_t}$) directions:

            $$h_t^{\text{BiLSTM}} = [\overrightarrow{h_t} \,\|\, \overleftarrow{h_t}] \in \mathbb{R}^{2 \cdot d_{\text{units}}}$$

            This enables the model to condition upon both preceding sentiment cues (*\"not bad...\"*) and subsequent qualifiers (*\"...for the price\"*).
            """
        )

    # UNIT 3
    with tab_u3:
        st.markdown("### ⚡ Unit 3: Transformer Architectures & Self-Attention")
        st.markdown(
            r"""
            #### 1. Scaled Dot-Product Attention
            Unlike sequential recurrence, Transformers compute all token pairwise dependencies in parallel using scaled dot-product self-attention:

            $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$

            Where query $Q$, key $K$, and value $V$ matrices are linear projections of input embeddings:

            $$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

            #### 2. Multi-Head Attention
            Multi-Head Attention projects representations into $h$ distinct representation subspaces:

            $$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$
            $$\text{where } \text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

            #### 3. BERT Fine-Tuning Paradigm
            BERT (Bidirectional Encoder Representations from Transformers) is pre-trained via Masked Language Modeling (MLM) and Next Sentence Prediction (NSP).
            For 3-class sentiment classification, the special `[CLS]` token embedding $h_{\text{[CLS]}}$ is fed to a classification head:

            $$\hat{y} = \text{softmax}(W_{\text{cls}} h_{\text{[CLS]}} + b_{\text{cls}})$$
            """
        )

    # UNIT 4
    with tab_u4:
        st.markdown("### 🔮 Unit 4: Generative AI, LLMs & Synthetic Data Synthesis")
        st.markdown(
            r"""
            #### 1. Autoregressive Language Modeling
            Generative LLMs model the joint probability distribution over token sequences by autoregressively decomposing the likelihood:

            $$P(w_1, w_2, \dots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \dots, w_{t-1})$$

            #### 2. In-Context Learning: Zero-Shot vs Few-Shot Prompting
            Rather than updating network parameters $\theta$, Generative models perform task conditioning purely in the input context window:
            - **Zero-Shot:** $\hat{y} \sim P(\text{Sentiment} \mid \text{Review}, \text{Instruction})$
            - **Few-Shot:** $\hat{y} \sim P(\text{Sentiment} \mid \text{Ex}_1, \text{Label}_1, \dots, \text{Ex}_k, \text{Label}_k, \text{Review})$

            #### 3. Generative Data Augmentation
            By prompting foundation models (Google Gemini) to sample from domain-conditioned distributions:

            $$X_{\text{synth}} \sim P_{\text{LLM}}(\text{Customer Review} \mid \text{Sentiment} = c, \text{Domain} = \text{Hospitality})$$

            We construct $N_{\text{synth}} = 500$ new training instances to balance minority classes and smooth decision boundaries for discriminative classifiers.
            """
        )

    # UNIT 5
    with tab_u5:
        st.markdown("### ⚖️ Unit 5: Responsible AI, Probability Calibration & Evaluation")
        st.markdown(
            r"""
            #### 1. Multiclass Brier Score (Strictly Proper Scoring Rule)
            The multiclass Brier score measures the mean squared difference between predicted probability vectors and one-hot true indicators:

            $$\text{Brier Score} = \frac{1}{N} \sum_{i=1}^N \sum_{k=0}^{K-1} (p_{ik} - y_{ik})^2, \quad \text{where } \text{BS} \in [0, 2]$$

            A lower Brier score guarantees both sharper accuracy and well-calibrated confidence.

            #### 2. Expected Calibration Error (ECE)
            Group predictions into $M$ equal-width confidence bins $B_m \subset (0, 1]$. ECE is the weighted average absolute difference between empirical accuracy and average confidence:

            $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

            #### 3. Platt Scaling (Sigmoid Multiclass Calibration)
            Raw classifier outputs $z_k$ are calibrated using learned scaling parameters $(A_k, B_k)$:

            $$p_k^{\text{calibrated}} = \frac{1}{1 + \exp(A_k z_k + B_k)}$$

            #### 4. Counterfactual Fairness & Demographic Parity
            A model satisfies counterfactual fairness with respect to cultural attribute $A$ if:

            $$P(\hat{Y} \mid X = x, A = a) = P(\hat{Y} \mid X = x, A = a')$$
            """
        )

