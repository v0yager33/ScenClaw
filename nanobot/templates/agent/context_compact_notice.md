[Context compacted] To stay within the context window, the earlier part of this task ({{ folded }} messages) was removed from the context and replaced by the summary below. The summary is lossy.
{% if transcript %}
The removed messages are saved verbatim at `{{ transcript }}` (JSONL, one message per line). Use grep or read_file on that file when you need exact tool output, code or error text that the summary leaves out.
{% endif %}
Continue the task from where it left off. Do not ask the user to repeat anything and do not redo work the summary reports as done.

<summary>
{{ summary }}
</summary>
{% if files %}

Files referenced by tool calls in the removed part (most recent last); read them again if you need their current content:
{% for path in files %}
- {{ path }}
{% endfor %}
{% endif %}
{% if user_messages %}

User messages in the removed part (verbatim, long ones clipped):
{% for text in user_messages %}
- {{ text }}
{% endfor %}
{% endif %}
