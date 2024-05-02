import tkinter as tk
from tkinter import filedialog, scrolledtext, Menu
from hybrid_cs_sa import HybridAlgorithm , ProgramAnalysis  # Uncomment when available
from copyright import LegalApp,  initialize_db
import time
legal_app_instance = None




def show_splash_screen(root, callback):
    splash = tk.Toplevel(root)
    splash.title("OOP Test case generator")
    splash.geometry("300x200")
    splash_label = tk.Label(splash, text="OOP Test case generator", font=("Helvetica", 18))
    splash_label.pack(expand=True)
    splash.overrideredirect(True)

    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x_coordinate = int((screen_width / 2) - (300 / 2))
    y_coordinate = int((screen_height / 2) - (200 / 2))
    splash.geometry(f"+{x_coordinate}+{y_coordinate}")

    # Ensure splash is destroyed before continuing
    splash.after(3000, lambda: [splash.destroy(), callback()])



def start_legal_app(root):
    global legal_app_instance  # Indicate that we're using the global variable
    # Ensure this function correctly initializes and displays LegalApp.
    print("Initializing LegalApp...")  # Debug print

    root.deiconify()  # Make the root window visible if it was hidden.
    legal_app = LegalApp(root, on_completion=on_legal_app_completion)
    return legal_app


# Somewhere else in your code, you set up the on_completion callback:
def on_legal_app_completion(root):
    initialize_main_application(root)  # This will set up the main application
    root.deiconify()  # or other logic to show the main application


def initialize_main_application(root):
    # First, clear out any existing widgets from the root or login frame
    for widget in root.winfo_children():
        widget.destroy()

    # Now, initialize the widgets for the main application.
    global code_input, results_display, generate_button, save_button
    code_input = scrolledtext.ScrolledText(root, height=15, width=80)
    code_input.pack()

    generate_button = tk.Button(root, text="Generate Test Cases", command=lambda:generate_test_cases(root))
    generate_button.pack()

    results_display = scrolledtext.ScrolledText(root, height=15, width=80)
    results_display.pack()

    save_button = tk.Button(root, text="Save Test Cases", command=save_test_cases)
    save_button.pack()

    # Inside initialize_main_application(root) function, add a new button for pytest templates
    template_button = tk.Button(root, text="View Pytest Templates", command=lambda: show_pytest_templates(root))
    template_button.pack()


    # Finally, make the root window visible if it was hidden
    initialize_menu_bar(root)


def initialize_menu_bar(root):
    menu_bar = Menu(root)

    # File menu
    file_menu = Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="File", menu=file_menu)
    file_menu.add_command(label="New File")  # Dummy command, replace with actual functionality
    file_menu.add_separator()
    file_menu.add_command(label="Open Folder")  # Dummy command
    file_menu.add_separator()
    file_menu.add_command(label="Save as")  # Dummy command

    # Add more menus as needed...

    # Guides menu
    guides_menu = Menu(menu_bar, tearoff=0)
    menu_bar.add_cascade(label="Guides", menu=guides_menu)
    guides_menu.add_command(label="Using Test Cases", command=show_guide_using_test_cases)
    guides_menu.add_separator()
    guides_menu.add_command(label="Understanding Fixtures", command=understanding_fixtures)  # Replace dummy_guide_command with actual
    guides_menu.add_command(label="Parameterized Tests", command=parametized_tests)
    guides_menu.add_command(label="Mocking in Tests", command=mocking_in_tests)

    root.config(menu=menu_bar)


def show_guide_using_test_cases():
    # Implement the guide content display. This could open a new window with the guide text, for example.
    guide_window = tk.Toplevel()
    guide_window.title("Guide: Using Test Cases")
    guide_text = tk.scrolledtext.ScrolledText(guide_window, wrap=tk.WORD, height=15, width=80)
    guide_text.pack(fill=tk.BOTH, expand=True)
    guide_content = """
Creating and using test cases effectively is a fundamental part of software development, ensuring your code behaves as expected and facilitating maintenance and updates. Here's a basic guide to using test cases, focusing on Python's pytest framework, which is commonly used for writing and running tests.
__________________________
Understanding Test Cases |
_________________________|
A test case is a set of conditions or variables under which a tester determines whether a system under test satisfies requirements and works correctly. In software testing, a test case might include specific inputs, execution conditions, and expected results.

____________________________
Getting Started with pytest|
___________________________|
pytest is a popular testing framework for Python that makes it easy to write simple tests, yet scales to support complex functional testing. It provides features like fixtures, parametrization, and plugins to extend its functionality.
________________
Installation   |
_______________|
To get started with pytest, you need to install it using pip:\n
                 ____________________
                 |pip install pytest|
                 |__________________|
\n
________________________
Writing Your First Test|
_______________________|
A pytest test case is a function defined in a Python file and named test_*.py or *_test.py. A simple test case can be viewed 
in the menu below.
\n
____________________
Test Organization  |
___________________|
As your project grows, organize your tests into separate files and directories. Common practices include:

Grouping related tests within the same file.
Using directories to separate tests for different modules or components.
\n
___________________
Best Practices    |
__________________|
1. Write Clear, Concise Tests: Each test should focus on a single functionality.
2. Use Descriptive Test Names: Test names should describe what they test.
3. Keep Tests Independent: Each test should set up its own data.
4. Use Fixtures for Common Test Objects: This reduces duplication and keeps tests clean.
5. Review Test Failures Carefully: When a test fails, understand why before fixing it. This might reveal underlying issues in your code.


"""
    guide_text.insert(tk.END, guide_content)
    guide_text.config(state=tk.DISABLED)

def understanding_fixtures():
    understanding_fixture_window = tk.Toplevel()
    understanding_fixture_window.title("Guide: Understanding fixtures")
    und_fix_text = tk.scrolledtext.ScrolledText(understanding_fixture_window, wrap=tk.WORD, height=15, width=80)
    und_fix_text.pack(fill=tk.BOTH, expand=True)
    fix_content = """\
    Fixtures in pytest are used to set up and tear down resources before and after tests, ensuring tests run under the conditions they require.

    Key Points:
    - Fixtures are defined using the @pytest.fixture decorator.
    - They can provide test data, instantiate objects, connect to databases, and more.
    - Tests request fixtures by including fixture names as parameters.
    - Fixtures can be scoped at different levels such as function, class, module, or session.

    Example:

    import pytest

    @pytest.fixture
    def sample_data():
        # Setup code here
        data = {'key': 'value'}
        return data

    def test_example(sample_data):
        # Test using fixture data
        assert sample_data['key'] == 'value'

    Fixtures are powerful tools for creating reusable, modular test components."""

    und_fix_text.insert(tk.END, fix_content)
    und_fix_text.config(state=tk.DISABLED)


def parametized_tests():
    parametized_test_window = tk.Toplevel()
    parametized_test_window.title("Guide: Parametized tests")
    param_test_text = tk.scrolledtext.ScrolledText(parametized_test_window, wrap=tk.WORD, height=15, width=80)
    param_test_text.pack(fill=tk.BOTH, expand=True)
    param_content = """\
    Parameterized tests allow you to run the same test function with different inputs, making it easy to test multiple cases with a single function.

    Key Points:
    - Use @pytest.mark.parametrize decorator to define multiple sets of arguments and values for a test function.
    - Reduces code duplication and increases test coverage.
    - Ideal for testing functions with various inputs and expected outputs.

    Example:

    import pytest

    @pytest.mark.parametrize("input,expected", [
        (2, 4),
        (3, 9),
        (4, 16),
    ])
    def test_square(input, expected):
        result = input * input
        assert result == expected

    This will run test_square three times with different input and expected values."""

    param_test_text.insert(tk.END, param_content)
    param_test_text.config(state=tk.DISABLED)

def mocking_in_tests():
    mocking_in_tests_window = tk.Toplevel()
    mocking_in_tests_window .title("Guide: Mocking in tests")
    mock_test_text = tk.scrolledtext.ScrolledText(mocking_in_tests_window, wrap=tk.WORD, height=15, width=80)
    mock_test_text.pack(fill=tk.BOTH, expand=True)
    param_content = """\
    Mocking is a technique used in testing to replace real objects in your system under test with mock objects that simulate the behavior of real objects.

    Key Points:
    - Useful in isolating tests from external dependencies or side effects.
    - Helps in testing a component in isolation from its integration layers or databases.
    - Python’s unittest.mock module provides a powerful framework for mocking.

    Example:

    from unittest.mock import MagicMock
    import pytest

    def external_dependency():
        # Function that would normally perform an external call
        pass

    def test_with_mock():
        with pytest.mock.patch('path.to.external_dependency', return_value='mocked value'):
            result = external_dependency()
            assert result == 'mocked value'

    Here, external_dependency is mocked to return 'mocked value' during the test."""
    mock_test_text.insert(tk.END, param_content)
    mock_test_text.config(state=tk.DISABLED)


def main():
    root = tk.Tk()
    root.title("Test Case Generator")
    root.geometry("800x600")  # Adjust the size as needed
    root.withdraw()  # Hide the main window


    # Initialize DB for LegalApp
    initialize_db()

    # Show splash screen first
    show_splash_screen(root, lambda: start_legal_app(root))




    root.mainloop()


def show_pytest_templates(parent_root):
    try:
        template_win = tk.Toplevel(parent_root)
        template_win.title("Pytest Templates")
        template_win.geometry("600x400")

        template_types = ["Basic Test", "Test Class", "Fixture", "Parametrized Test", "Conftest", "Mocking"]
        selected_template = tk.StringVar(template_win)
        selected_template.set(template_types[0])  # default value

        def update_template_display(choice):
            try:
                print(f"Updating template display for: {choice}")  # Debug print
                template_display.config(state=tk.NORMAL)
                template_display.delete('1.0', tk.END)
                template_content = get_template_content(choice)
                template_display.insert(tk.END, template_content)
                template_display.config(state=tk.DISABLED)
            except Exception as e:
                print(f"Error in update_template_display: {e}")

        template_display = scrolledtext.ScrolledText(template_win, height=20, width=70)
        template_display.pack()

        dropdown = tk.OptionMenu(template_win, selected_template, *template_types, command=update_template_display)
        dropdown.pack()

        update_template_display(template_types[0])
    except Exception as e:
        print(f"Error in show_pytest_templates: {e}")

def get_template_content(choice):
    templates = {
        "Basic Test": """\
import pytest

def test_function():
    assert True""",
        "Test Class": """\
import pytest

class TestClass:
    def test_method(self):
        assert True""",
        "Fixture": """\
import pytest

@pytest.fixture
def sample_fixture():
    return 'sample data'

def test_with_fixture(sample_fixture):
    assert sample_fixture == 'sample data'""",
        "Parametrized Test": """\
import pytest

@pytest.mark.parametrize('input,expected', [
    (1, 2),
    (2, 3),
    (3, 4)
])
def test_increment(input, expected):
    assert input + 1 == expected""",
        "Conftest": """\
# This would go in a conftest.py file

import pytest

@pytest.fixture
def fixture_available_across_tests():
    return 'data'""",
        "Mocking": """\
import pytest
from unittest.mock import MagicMock

def test_mocking():
    mock = MagicMock()
    mock.return_value = 'mocked value'
    assert mock() == 'mocked value'"""
    }
    return templates.get(choice, "Template not found.")


def generate_test_cases(root):
    program_code = code_input.get("1.0", tk.END)
    start_time = time.perf_counter()  # Start timing before the operation
    hybrid_tool = HybridAlgorithm(user_input_code=program_code)

    # Supposedly, these functions generate and return the test cases and coverage report
    results = hybrid_tool.generate_test_cases(program_code)
    test_cases = hybrid_tool.generate_test_cases(program_code)
    # print(f"Generated test cases: {test_cases}")

    program_analysis = ProgramAnalysis(program_code)
    program_analysis.extract_structure()
    coverage_report = program_analysis.generate_coverage_report()

    end_time = time.perf_counter()  # End timing after the operation
    execution_time = end_time - start_time  # Calculate the execution time

    # Schedule display_results to run on the main thread
    root.after(0, lambda: display_results(root, test_cases, coverage_report, execution_time))


def display_results(root, test_cases, coverage_report, execution_time):
    print(f"Coverage report: {coverage_report}")

    # Make sure this runs on the main thread if called from another thread
    formatted_test_cases = format_test_cases(test_cases)
    formatted_coverage_report = format_coverage_report(coverage_report)

    results_display.config(state=tk.NORMAL)  # enable editing of the widget
    results_display.delete('1.0', tk.END)  # clear current contents
    results_display.insert(tk.END, "Generated Test Cases:\n" + formatted_test_cases)
    results_display.insert(tk.END, f"Execution Time: {execution_time} seconds\n\n")  # Display execution time
    results_display.insert(tk.END, "\nCoverage Report:\n" + formatted_coverage_report)
    results_display.config(state=tk.DISABLED)  # disable editing of the widget

def format_test_cases(test_cases):
    formatted_output = ""
    for i, test_case in enumerate(test_cases, 1):
        formatted_output += f"Test Case {i}:\n"
        for key, value in test_case.items():
            formatted_output += f"    {key}: {value}\n"
        formatted_output += "\n"
    return formatted_output


def format_coverage_report(coverage_report):
    formatted_report = f"Coverage Percentage: {coverage_report['coverage_percentage']:.2f}%\n"
    formatted_report += f"Covered Methods Count: {coverage_report['covered_methods']}\n"
    formatted_report += f"Uncovered Methods Count: {coverage_report['uncovered_methods']}\n"
    return formatted_report





def save_test_cases():
    file_path = filedialog.asksaveasfilename(defaultextension=".txt")
    if file_path:
        with open(file_path, "w") as file:
            file.write(results_display.get("1.0", tk.END))

if __name__ == "__main__":
    main()