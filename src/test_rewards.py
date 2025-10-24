import re

def match(sol):
    sol_match = re.search(r'\\boxed\{(.*?)}\s*', sol)
    ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()

    print(ground_truth)

def match_all(content):
    print(re.findall(r'\\boxed\{(.*?)}\s{,3}$', content, re.DOTALL))

if __name__ == "__main__":
    for sol in ["The solution is \\boxed{1} The solution is \\boxed{2}",
                "The solution is The solution is \\boxed{1}",
                "The solution is \\boxed{1}  \n "]:

        if sol.count("\\boxed{") != 1:
            print("invalid")
        else:
            match_all(sol)

    # Using .format()
    template = "{Input} Reason with the visual information step by step, and put your final answer within \\boxed{{}}."
    result = template.format(Input="your_input_value")
    print(result)

    # Using f-strings
    input_value = "your_input_value"
    result = f"{input_value} Reason with the visual information step by step, and put your final answer within \\boxed{{}}."
    print(result)

    # Alternative with .format() using positional argument
    template = "{} Reason with the visual information step by step, and put your final answer within \\boxed{{}}."
    result = template.format("your_input_value")
    print(result)


