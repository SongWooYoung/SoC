# Qwen3.5 Chat Template & Inference Settings

Source: https://huggingface.co/Qwen/Qwen3.5-4B

---

## 1. Thinking vs Non-Thinking Mode

Qwen3.5 **thinks by default** — it generates `<think>\n...\n</think>\n\n` before the final response.

**Qwen3.5 does NOT support the Qwen3 soft-switch (`/think`, `/nothink`).** Mode is controlled entirely via the `enable_thinking` parameter at generation time.

### Disable thinking (Non-Thinking / Instruct mode)

The chat template checks `enable_thinking`:
- When `enable_thinking` is `false`, the generation prompt is:
  ```
  <|im_start|>assistant\n<think>\n\n</think>\n\n
  ```
  (empty think block — model skips reasoning and gives direct answer)

- When `enable_thinking` is `true` (default), the prompt is:
  ```
  <|im_start|>assistant\n<think>\n
  ```
  (model generates reasoning, then `</think>\n\n`, then final answer)

#### API usage
```python
# Via vLLM/SGLang OpenAI-compatible API:
extra_body={"chat_template_kwargs": {"enable_thinking": False}}

# Via Alibaba Cloud Model Studio (DashScope):
extra_body={"enable_thinking": False}
```

---

## 2. Recommended Sampling Parameters

| Mode | Task | temp | top_p | top_k | min_p | presence_penalty | repetition_penalty |
|------|------|------|-------|-------|-------|------------------|-------------------|
| Thinking | General | 1.0 | 0.95 | 20 | 0.0 | 1.5 | 1.0 |
| Thinking | Precise coding (WebDev) | 0.6 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking | General | 0.7 | 0.8 | 20 | 0.0 | 1.5 | 1.0 |
| Non-thinking | Reasoning | 1.0 | 1.0 | 40 | 0.0 | 2.0 | 1.0 |

- `max_tokens`: 32768 for most queries, 81920 for complex math/code competition problems
- `presence_penalty` 0–2 helps reduce endless repetitions (but high values may cause language mixing)

---

## 3. Chat Template (Jinja2)

Format: ChatML with `<|im_start|>role\n...<|im_end|>\n`

### Basic structure
```
<|im_start|>system\n{system_message}<|im_end|>\n
<|im_start|>user\n{user_message}<|im_end|>\n
<|im_start|>assistant\n<think>\n{reasoning}</think>\n\n{response}<|im_end|>\n
```

### With thinking disabled
```
<|im_start|>system\n{system_message}<|im_end|>\n
<|im_start|>user\n{user_message}<|im_end|>\n
<|im_start|>assistant\n<think>\n\n</think>\n\n{response}<|im_end|>\n
```

### Vision tokens
- Image: `<|vision_start|><|image_pad|><|vision_end|>`
- Video: `<|vision_start|><|video_pad|><|vision_end|>`
- Optional labeling with `add_vision_id`: `Picture 1: <|vision_start|>...`

### Multi-turn: thinking content stripped from history
In multi-turn conversations, historical assistant responses should only include the final output (no `<think>...</think>` block). The Jinja2 template handles this automatically. For non-Jinja frameworks, developers must strip thinking content manually.

---

## 4. Special Token IDs

| Token | ID | Purpose |
|-------|------|---------|
| `<\|endoftext\|>` | 248044 | EOS |
| `<\|im_start\|>` | 248045 | ChatML role start |
| `<\|im_end\|>` | 248046 | ChatML role end |
| `<\|object_ref_start\|>` | 248047 | Object reference start |
| `<\|object_ref_end\|>` | 248048 | Object reference end |
| `<\|box_start\|>` | 248049 | Bounding box start |
| `<\|box_end\|>` | 248050 | Bounding box end |
| `<\|quad_start\|>` | 248051 | Quad start |
| `<\|quad_end\|>` | 248052 | Quad end |
| `<\|vision_start\|>` | 248053 | Vision content start |
| `<\|vision_end\|>` | 248054 | Vision content end |
| `<\|vision_pad\|>` | 248055 | Vision padding |
| `<\|image_pad\|>` | 248056 | Image placeholder (config: `image_token_id`) |
| `<\|video_pad\|>` | 248057 | Video placeholder (config: `video_token_id`) |

---

## 5. Tool Calling Format

Tools are injected in the system message:
```
<|im_start|>system
# Tools

You have access to the following functions:

<tools>
{tool_json_1}
{tool_json_2}
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=param_name>
value
</parameter>
</function>
</tool_call>
<|im_end|>
```

Tool responses are wrapped in `<tool_response>...</tool_response>` inside user messages.

---

## 6. Full Jinja2 Chat Template
 
 - For non-thinking, add this to the top of jinja2 template
   {% set enable_thinking = false %} 

```jinja2
{%- set image_count = namespace(value=0) %}
{%- set video_count = namespace(value=0) %}
{%- macro render_content(content, do_vision_count, is_system_content=false) %}
    {%- if content is string %}
        {{- content }}
    {%- elif content is iterable and content is not mapping %}
        {%- for item in content %}
            {%- if 'image' in item or 'image_url' in item or item.type == 'image' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain images.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set image_count.value = image_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Picture ' ~ image_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|image_pad|><|vision_end|>' }}
            {%- elif 'video' in item or item.type == 'video' %}
                {%- if is_system_content %}
                    {{- raise_exception('System message cannot contain videos.') }}
                {%- endif %}
                {%- if do_vision_count %}
                    {%- set video_count.value = video_count.value + 1 %}
                {%- endif %}
                {%- if add_vision_id %}
                    {{- 'Video ' ~ video_count.value ~ ': ' }}
                {%- endif %}
                {{- '<|vision_start|><|video_pad|><|vision_end|>' }}
            {%- elif 'text' in item %}
                {{- item.text }}
            {%- else %}
                {{- raise_exception('Unexpected item type in content.') }}
            {%- endif %}
        {%- endfor %}
    {%- elif content is none or content is undefined %}
        {{- '' }}
    {%- else %}
        {{- raise_exception('Unexpected content type.') }}
    {%- endif %}
{%- endmacro %}
{%- if not messages %}
    {{- raise_exception('No messages provided.') }}
{%- endif %}
{%- if tools and tools is iterable and tools is not mapping %}
    {{- '<|im_start|>system\n' }}
    {{- "# Tools\n\nYou have access to the following functions:\n\n<tools>" }}
    {%- for tool in tools %}
        {{- "\n" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- "\n</tools>" }}
    {{- '\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\nthat can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n- Required parameters MUST be specified\n- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n</IMPORTANT>' }}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {%- if content %}
            {{- '\n\n' + content }}
        {%- endif %}
    {%- endif %}
    {{- '<|im_end|>\n' }}
{%- else %}
    {%- if messages[0].role == 'system' %}
        {%- set content = render_content(messages[0].content, false, true)|trim %}
        {{- '<|im_start|>system\n' + content + '<|im_end|>\n' }}
    {%- endif %}
{%- endif %}
{%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
{%- for message in messages[::-1] %}
    {%- set index = (messages|length - 1) - loop.index0 %}
    {%- if ns.multi_step_tool and message.role == "user" %}
        {%- set content = render_content(message.content, false)|trim %}
        {%- if not(content.startswith('<tool_response>') and content.endswith('</tool_response>')) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if ns.multi_step_tool %}
    {{- raise_exception('No user query found in messages.') }}
{%- endif %}
{%- for message in messages %}
    {%- set content = render_content(message.content, true)|trim %}
    {%- if message.role == "system" %}
        {%- if not loop.first %}
            {{- raise_exception('System message must be at the beginning.') }}
        {%- endif %}
    {%- elif message.role == "user" %}
        {{- '<|im_start|>' + message.role + '\n' + content + '<|im_end|>' + '\n' }}
    {%- elif message.role == "assistant" %}
        {%- set reasoning_content = '' %}
        {%- if message.reasoning_content is string %}
            {%- set reasoning_content = message.reasoning_content %}
        {%- else %}
            {%- if '</think>' in content %}
                {%- set reasoning_content = content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
                {%- set content = content.split('</think>')[-1].lstrip('\n') %}
            {%- endif %}
        {%- endif %}
        {%- set reasoning_content = reasoning_content|trim %}
        {%- if loop.index0 > ns.last_query_index %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content + '\n</think>\n\n' + content }}
        {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
        {%- endif %}
        {%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}
            {%- for tool_call in message.tool_calls %}
                {%- if tool_call.function is defined %}
                    {%- set tool_call = tool_call.function %}
                {%- endif %}
                {%- if loop.first %}
                    {%- if content|trim %}
                        {{- '\n\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- else %}
                        {{- '<tool_call>\n<function=' + tool_call.name + '>\n' }}
                    {%- endif %}
                {%- else %}
                    {{- '\n<tool_call>\n<function=' + tool_call.name + '>\n' }}
                {%- endif %}
                {%- if tool_call.arguments is defined %}
                    {%- for args_name, args_value in tool_call.arguments|items %}
                        {{- '<parameter=' + args_name + '>\n' }}
                        {%- set args_value = args_value | tojson | safe if args_value is mapping or (args_value is sequence and args_value is not string) else args_value | string %}
                        {{- args_value }}
                        {{- '\n</parameter>\n' }}
                    {%- endfor %}
                {%- endif %}
                {{- '</function>\n</tool_call>' }}
            {%- endfor %}
        {%- endif %}
        {{- '<|im_end|>\n' }}
    {%- elif message.role == "tool" %}
        {%- if loop.previtem and loop.previtem.role != "tool" %}
            {{- '<|im_start|>user' }}
        {%- endif %}
        {{- '\n<tool_response>\n' }}
        {{- content }}
        {{- '\n</tool_response>' }}
        {%- if not loop.last and loop.nextitem.role != "tool" %}
            {{- '<|im_end|>\n' }}
        {%- elif loop.last %}
            {{- '<|im_end|>\n' }}
        {%- endif %}
    {%- else %}
        {{- raise_exception('Unexpected message role.') }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
    {%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\n\n</think>\n\n' }}
    {%- else %}
        {{- '<think>\n' }}
    {%- endif %}
{%- endif %}
```

---

## 7. Key Implementation Notes for C++ Inference

### Token sequence for generation prompt (thinking mode)
```
[<|im_start|>] "assistant\n<think>\n"
```
Token IDs: `248045`, then encode `"assistant\n<think>\n"`

### Token sequence for generation prompt (non-thinking mode)
```
[<|im_start|>] "assistant\n<think>\n\n</think>\n\n"
```
The model will immediately produce the final answer without reasoning.

### Parsing output
1. Model generates `<think>\n{reasoning}\n</think>\n\n{final_answer}`
2. For display: strip everything between `<think>` and `</think>` inclusive
3. For multi-turn history: only keep the `{final_answer}` part

### Context length
- Native: 262,144 tokens
- Extended with YaRN: up to 1,010,000 tokens
- Minimum recommended for thinking: 128K tokens

### YaRN config for extended context
```json
{
    "mrope_interleaved": true,
    "mrope_section": [11, 11, 10],
    "rope_type": "yarn",
    "rope_theta": 10000000,
    "partial_rotary_factor": 0.25,
    "factor": 4.0,
    "original_max_position_embeddings": 262144
}
```
