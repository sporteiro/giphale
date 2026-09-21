CRITICAL: You MUST use Anthropic-style XML tags for ALL tool invocations.
DO NOT output JSON blocks like {"name": "write_to_file"}.
DO NOT output JSON blocks like {"name": "execute_command"}.
DO NOT output JSON blocks like {"name": "read_file"}.
You MUST strictly use XML tags:
<write_to_file><path>...</path><content>...</content></write_to_file>
<execute_command><command>...</command></execute_command>
<read_file><path>...</path></read_file>
<replace_in_file><path>...</path><diff>...</diff></replace_in_file>
<list_files><path>...</path></list_files>
<search_files><path>...</path><regex>...</regex></search_files>
