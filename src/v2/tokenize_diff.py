import tokenize
import io
#Note: Indentation was normalized into four space levels. Each partial level is rounded down

def indentation_level(line):
    tabs = line.count('\t')
    level = (len(line) - len(line.lstrip(' '))) // 4
    return f'<INDENT_{tabs + level}>'


def tokenize_diff(changes):
    tokens = []
    lines = changes.splitlines()

    for line in lines:
        line_tokens = []
        if not line:
            continue
        elif not line.startswith(('+ ', '- ')):
            continue

        holder = ''
        if line[0] == '-':
            holder = '<DELETE>'
            line = line[2:]
        elif line[0] == '+':
            holder = '<ADD>'
            line = line[2:]
        print(line)
        indentation = indentation_level(line)
        line = line.strip() #strip after getting the indent level
        
        try:
            token_generator = tokenize.generate_tokens(io.StringIO(line).readline)
            for token in token_generator:
                if token.type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL, tokenize.COMMENT):
                    continue
                line_tokens.append(token.string)
            if holder and line_tokens:
                tokens.append(holder)
                tokens.append(indentation)
            tokens += line_tokens
        except tokenize.TokenError:
            pass

    return tokens