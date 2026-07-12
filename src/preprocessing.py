from difflib import ndiff

def generate_diff(buggy_code, fixed_code): #goal is to create the difference between buggy vs fixed
    #design decisions:
    #1. remove the visual ? and lines where no changes to reduce noise
    buggy_code = buggy_code.splitlines() #splits based off of lines for line diff 
    fixed_code = fixed_code.splitlines()
    diff = list(ndiff(buggy_code, fixed_code))
    diff = [line for line in diff if line.startswith('+') or line.startswith('-')]
    diff = '\n'.join(diff)
    return diff

