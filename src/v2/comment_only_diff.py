def is_comment_only(diff): #returns true or false depending on if the comment only contains comments
    lines = diff.splitlines()

    for line in lines:
        if not line.startswith(('+','-')):
            continue
        else:
            line = line.replace(line[0], '', 1)
        line = line.strip()
        if not line:
            continue
        if not line.startswith('#'):
            return False
    return True