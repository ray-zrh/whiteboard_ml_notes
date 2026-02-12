
import os
import re
import glob

def get_indent(line):
    return len(line) - len(line.lstrip())

def process_file(filepath):
    print(f"Processing {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []

    # State
    list_indent_stack = [] # Stack of indentations for nested lists
    in_math = False
    math_target_indent = 0

    # Regex
    list_item_re = re.compile(r'^(\s*)([-*+]|\d+\.)\s+')
    block_math_re = re.compile(r'^(\s*)\$\$\s*$')

    for i, line in enumerate(lines):
        # 1. Handle Math Block
        math_match = block_math_re.match(line)
        if math_match:
            current_indent = len(math_match.group(1))

            if not in_math:
                # Math Block Start
                # Determine target indentation
                target_indent = current_indent

                # Check if we are inside a list context
                # and if the current indent is "too shallow" (<= last list item indent)
                if list_indent_stack:
                    last_list_indent = list_indent_stack[-1]
                    # If math is at same level or less than list item, it breaks the list.
                    # It should be indented by +4 (or +2/3 depending on style, but 4 is safe).
                    if current_indent <= last_list_indent:
                        target_indent = last_list_indent + 4

                math_target_indent = target_indent
                new_lines.append(" " * math_target_indent + "$$\n")
                in_math = True
            else:
                # Math Block End
                new_lines.append(" " * math_target_indent + "$$\n")
                in_math = False
            continue

        # 2. Handle Content inside Math Block
        if in_math:
            # Strip and re-indent to match the target
            stripped = line.strip()
            if stripped:
                new_lines.append(" " * math_target_indent + stripped + "\n")
            else:
                new_lines.append("\n") # Keep blank lines in math? Usually simplify.
            continue

        # 3. Handle Normal Text / List Items

        # Check if empty line
        if not line.strip():
            new_lines.append(line)
            continue

        curr_indent = get_indent(line)

        # Check if List Item
        list_match = list_item_re.match(line)
        if list_match:
            # It is a list item
            # Update stack: if indent > last, push. If indent <= last, pop until correct level.
            # Simplified: just keep track of "current active list indentation"

            # Logic:
            # If indent > last, add to stack (nested) - but only if it makes sense?
            # Actually for this purpose, we just need to know "what lies above".
            # If we see a list item at indent X, then any subsequent block math should be > X.

            # If new list item is shallower or equal, pop deeper indents
            while list_indent_stack and curr_indent < list_indent_stack[-1]:
                list_indent_stack.pop()

            if not list_indent_stack or curr_indent > list_indent_stack[-1]:
                list_indent_stack.append(curr_indent)
            elif curr_indent == list_indent_stack[-1]:
                pass # Same level sibling

            new_lines.append(line)

        else:
            # Normal text line (not list item, not math)
            # Check if it exits the list
            # If text is unindented (indent 0) and we have a list stack, it likely closes the list
            if curr_indent == 0 and list_indent_stack:
                list_indent_stack = []

            # Also, if text is less indented than current list level, it might close levels
            while list_indent_stack and curr_indent < list_indent_stack[-1]:
               list_indent_stack.pop()

            new_lines.append(line)

    # Check if changed
    new_content = "".join(new_lines)
    original_content = "".join(lines)

    if new_content != original_content:
        print(f"Fixed {filepath}")
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_content)
    else:
        print(f"No changes for {filepath}")

files = glob.glob('/Users/ruihaozhang/Documents/code/whiteboard_ml_notes/notes/chapters/*.md')
for f in files:
    process_file(f)
