import os

file_path = "ui/app.py"

if os.path.exists(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    fixed = False
    for i, line in enumerate(lines):
        # Find the line causing the SyntaxError
        if "nonlocal metadata_collected" in line:
            # Trace backwards to find the nearest nested function definition
            for j in range(i, -1, -1):
                if lines[j].lstrip().startswith("def "):
                    # Get the exact indentation level of the function
                    indent = lines[j][:len(lines[j]) - len(lines[j].lstrip())]
                    
                    # Inject the missing variable binding right above the function
                    if j > 0 and "metadata_collected =" not in lines[j-1]:
                        lines.insert(j, indent + "metadata_collected = {}\n")
                        fixed = True
                    break
            break

    if fixed:
        with open(file_path, "w", encoding="utf-8") as f:
            f.writelines(lines)
        print("✅ FIXED: 'metadata_collected' scope binding successfully added to ui/app.py!")
    else:
        print("⚠️ Variable already bounded or pattern not found.")
else:
    print(f"❌ Error: {file_path} not found. Ensure you are running this in the RAG root folder.")