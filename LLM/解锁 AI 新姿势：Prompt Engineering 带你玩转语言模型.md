## 解锁 AI 新姿势：Prompt Engineering 带你玩转语言模型！

想要让 AI 乖乖听话，成为你的专属文案助手、代码生成器、创意灵感库？🤫  答案就在 **Prompt Engineering（提示工程）** ！

###  🪄  Prompt Engineering：唤醒 AI 潜力的魔法棒

想象一下，你正在跟一位“AI 大师”交流。这位大师拥有无穷的知识和创造力，但ta需要你用**精准的语言**来引导ta，才能发挥出真正的实力。

Prompt Engineering 就是这样一门艺术： **通过设计和优化输入文本（Prompt），引导 AI 模型生成我们想要的结果。** 简单来说，就是用“提示词”跟 AI 模型对话，让ta明白你的需求！

###  🧱  base LLM：AI 大师的“基本功”

在正式开始“调教” AI 之前，我们需要先了解一下**基础模型**的概念。

基础模型就像 AI 大师的“基本功”，是通过海量数据训练出来的“武林高手”，掌握了语言的基本规律和模式。常见的  LLMs，例如 GPT-3、BERT 等，都属于基础模型。

###  💪  Instruction-Tuned LLM：让 AI 更“听话”

虽然基础模型已经拥有了强大的能力，但ta们有时会像“脱缰的野马”一样，难以控制。这时候就需要 **指令微调** 。\
大白话说，**就像你教一个孩子如何完成一个任务！** , 就像用各种例子教小朋友："小明，你看这句话：'今天天气真好！' 是不是感觉很开心？这就叫正面情绪！" 模型听得多了，自然就懂了。

> They begin with a base LLM and are **fine-tuned with input-output pairs** that include instructions and attempts to follow those instructions.\
> **Reinforcement Learning from Human Feedback (RLHF)** is often employed to refine the model further, making it better at being helpful, honest, and harmless.

#### 那怎么用RLHF呢？

1. 训练奖励模型 (Reward Model):\
收集人类对模型输出的评价数据，例如：哪些输出好 👍，哪些不好 👎。\
用这些数据训练一个“奖励模型”，它能自动判断模型输出的质量高低，就像教练给选手打分一样。

2. 强化学习优化模型:\
让模型根据新的指令生成多个输出结果。\
用奖励模型对这些结果进行评分。\
利用强化学习算法 (例如 PPO)，根据分数调整模型参数，让模型学会生成更高分的输出。

指令微调就像给 AI 大师制定了一套“行为准则”，让ta们更“听话”，更能理解人类的指令。通过指令微调，我们可以教会 AI 模型：

*   **理解不同类型的指令**:  例如生成文本、翻译语言、回答问题等。
*   **遵循特定的格式要求**:  例如输出文本的长度、风格、语气等。
*   **避免生成不安全或不合适的内容**。

###  ✍️  Instruction Prompt 的秘诀：让 AI 秒懂你的心

想要让 AI 模型完美执行你的指令，一份精心设计的 Prompt 必不可少！以下是一些构建提示的秘诀：

1. **Using Examples**:  开门见山，直接告诉 AI 你想要什么！例如：

    | Learning Type | Prompt (Input)                                                                                                                                        | Completion (Output)         |
    | :------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------- |
    | Zero-shot     | "The Sun is Shining". Translate to Spanish                                                                                                            | "El Sol está brillando".    |
    | One-shot      | "The Sun is Shining" => ""El Sol está brillando". <br> "It's a Cold and Windy Day" =>                                                                 | "Es un día frío y ventoso". |
    | Few-shot      | The player ran the bases => Baseball <br/> The player hit an ace => Tennis <br/> The player hit a six => Cricket <br/> The player made a slam-dunk => | Basketball                  |
    |               |       

2. **Prompt Cues**:  通过提供一些关键词 比如帮GPT起个头，引导 AI 生成你想要的内容。例如：

    | Number of Cues | Prompt (Input)                                                                                                                                                                                                                                                                                                                                                                                                                                               | Completion (Output)                                                                                                                                                                                                                                                                                       |
    | :------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
    | 0              | Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with ... since before recorded history. <br/> <br/>**Summarize This**                                       | Jupiter is the largest planet in our Solar System and the fifth one from the Sun. It is a gas giant with a mass 1/1000th of the Sun's, but it is heavier than all the other planets combined. Ancient civilizations have known about Jupiter for a long time, and it is easily visible in the night sky.. |
    | 1              | Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with ... since before recorded history. <br/> <br/>**Summarize This** <br/> What we learned is that Jupiter | is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with a mass one-thousandth that of the Sun, but two-and-a-half times that of all the other planets combined. It is easily visible to the naked eye and has been known since ancient times.                        |
    | 2              | Jupiter is the fifth planet from the Sun and the largest in the Solar System. It is a gas giant with ... since before recorded history. <br/> <br/>**Summarize This** <br/> Top 3 Facts We Learned:         | 1. Jupiter is the fifth planet from the Sun and the largest in the Solar System. <br/> 2. It is a gas giant with a mass one-thousandth that of the Sun...<br/> 3. Jupiter has been visible to the naked eye since ancient times ...                                                                       |
    |                |                                                                                                                                                                                                                                                                                                                                                                                                                                                              |                                                                                                                                                                                                                                                                                                           |

3. **Prompt Templates**:  预先设计好 Prompt 的结构和格式，方便重复使用。例如：

    ```
    请用简洁的语言解释以下概念：{}

    # 代码示例如下
    
    from langchain import PromptTemplate
    template = """\
    您是新公司的命名顾问。
    生产{product}的公司起什么好名字?
    """
    
    prompt = PromptTemplate.from_template(template)
    prompt.format(product="彩色袜子")

    # 输出
    您是新公司的命名顾问。
    一家生产彩色袜子的公司起什么名字好呢？

    
    ```

4. **Supporting Content**:   提供更多上下文信息，例如背景知识（knowledge）、相关示例等，帮助 AI 更准确地理解你的需求。\
   常常在制作自己的bot，通过定义bot的**身份；能力；任务**等要求，同时上传**知识库**，完成构建，比如chatGLM中的bot构建方式。例如：

    ```
    背景：我正在学习人工智能的相关知识。
    任务：请用通俗易懂的语言解释什么是机器学习。
    ```

### 🚀  Prompt Engineering：通往 AI 世界的钥匙

想要了解更多关于 Prompt Engineering 的知识？赶紧关注我们，一起探索 AI 世界的奥秘吧！
