from difflib import ndiff

def generate_diff(buggy_code, fixed_code): #goal is to create the difference between buggy vs fixed
    #design decisions:
    #1. remove the visual ? and lines where no changes to reduce noise
    buggy_lines = buggy_code.splitlines() #splits based off of lines for line diff 
    fixed_lines = fixed_code.splitlines()
    diff = list(ndiff(buggy_lines, fixed_lines))
    changed_lines = [line for line in diff if line.startswith('+') or line.startswith('-')]
    return '\n'.join(changed_lines)