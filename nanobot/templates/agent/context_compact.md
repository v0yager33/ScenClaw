You are compacting the working context of an agent that is in the middle of a long task. The messages below will be removed from its context and replaced by your summary, so the agent must be able to continue the task from your summary and the most recent messages alone.

The input may start with a <previous_summary>, which already covers everything before it, followed by the <messages> to fold in. Long tool results and arguments are clipped; the complete messages are kept in a transcript on disk.

First, inside <analysis></analysis>, go through the messages in order and note the user's requests, the approach taken, key decisions, files and identifiers involved, errors and how they were fixed, and what was being done most recently.

Then write the summary inside <summary></summary> with these sections:
1. Primary request and intent: everything the user asked for, in detail, including constraints and preferences.
2. Key technical concepts: technologies, conventions and facts that matter for the rest of the task.
3. Files and code: files read, created or modified, why each matters, exact paths, and short snippets only where the exact text matters.
4. Errors and fixes: every error met, how it was fixed, and any user feedback about it.
5. Problem solving: what has been solved and what investigation is still open.
6. User messages: every user message that is not a tool result, close to verbatim.
7. Pending tasks: work the user asked for that is not done yet.
8. Current work: exactly what was being worked on right before this point, with file names and code where relevant.
9. Next step: the step that directly continues the current work, quoting the latest instruction where it applies. Leave it empty if the task is finished.

Merge the previous summary into the new one and keep its facts unless later messages supersede them. Keep exact values such as paths, identifiers, numbers, commands and error messages. Do not add anything that is not in the input. Write in the language the user writes in. Keep the summary under about {{ max_words }} words.
