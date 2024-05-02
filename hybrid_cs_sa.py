"""
Hybrid Cuckoo search and simulated annealing algorithm for Test Case Generation

This code implements a hybrid of the cuckoo algorithm and simulated annealing algorithm to generate test cases for program analysis.

References:
- BrianMburu. (2023). Cockoo-Search-Algorithm. Github. https://github.com/BrianMburu/Cockoo-Search-Algorithm/blob/master/cuckoo_search.py
- Chncyhn. (2018). simulated-annealing-tsp. Github. https://github.com/chncyhn/simulated-annealing-tsp/blob/master/anneal.py
- Jedrazb. (2017). python-tsp-simulated-annealing. Github. https://github.com/jedrazb/python-tsp-simulated-annealing/blob/master/simulated_annealing.py
- Jifeng Wu. (2023). static-import-analysis-for-pure-python-projects. Github. https://github.com/abbaswu/static-import-analysis-for-pure-python-projects/tree/main/static_import_analysis
- Boadzie, D. (2023) Introduction to Abstract Syntax Trees in Python, Earthly Blog. Available at: https://earthly.dev/blog/python-ast/.
"""



import random
import math
from scipy.special import gamma
import numpy as np
import ast
import time


# Define Cuckoo Search Algorithm Components


class CuckooSearch:
    def __init__(self, population_size, nest_size, pa, beta=1.0):
        """
        Initialize parameters for Cuckoo Search

        :param population_size: Number of nests (solutions)
        :param nest_size: Dimension of the problem
        :param pa: Probability of discovering a worse nest
        :param beta: Parameter for Lévy flight
        """
        self.population_size = population_size
        self.nest_size = nest_size
        self.pa = pa
        self.beta = beta
        self.nests = np.random.rand(self.population_size, self.nest_size)
        self.best_nest = None
        self.best_fitness = float('inf')

    def generate_initial_population(self):
        """
        Generate initial population of solutions (nests)
        """
        self.nests = np.random.rand(self.population_size, self.nest_size)

    def get_fitness(self, nest):
        """
        Calculate the fitness of a nest (test case).

        :param nest: A dictionary representing a test case.
        :return: The fitness score of the test case.
        """
        try:
            return self.fitness(nest)
        except KeyError:
            # This Handles missing data keys in the test case
            return float('inf')

    def fitness(self, test_case):
# minimum threshold for the fitness score is set here ensuring that the fitness score never drops below this threshold.
        min_fitness_score = 0.001
# This is the starting score from which deductions or additions are made based on the evaluation of the test case.
        base_score = 0.5
        # The method retrieves values from the test case using test_case.get
        # check for missing keys and assign default values if necessary
        classes = test_case.get('classes', [])
        subclass = test_case.get('subclass', None)
        superclass = test_case.get('superclass', None)
        class_reference = test_case.get('class_reference', None)
        # Evaluate positive aspects of the test case
        polymorphism_score = self.evaluate_polymorphism(classes, test_case['method_name'])
        method_overriding_score = self.evaluate_method_overriding(subclass, superclass, test_case['method_name'])
        encapsulation_score = self.evaluate_encapsulation(class_reference) * 0.5
        inheritance_score = self.evaluate_inheritance(class_reference) * 0.5
        # Sum positive aspects to form the base evaluation score
        evaluation_score = polymorphism_score + method_overriding_score + encapsulation_score + inheritance_score
        # Evaluate potential faults and reduce the evaluation score accordingly
        fault_penalty = self.evaluate_faults(test_case)
        """
        random_variation
        introduces a random factor to the fitness score, simulating variability and unpredictability in the fitness 
        landscape, which helps in mimicking real-world scenarios where not all factors are predictable.
        """
        random_variation = random.uniform(-0.5, 0.5)  # Adjust the range as needed to create impact of randomness
        evaluation_score += random_variation
        # Ensure a minimum fitness score even in the presence of faults
        # The max function ensures that the score does not fall below the minimum fitness score set at the beginning.
        final_score = max(base_score - evaluation_score + fault_penalty, min_fitness_score)
        return final_score
    def evaluate_polymorphism(self, classes, method_name):
        behaviors = set()
        for cls in classes:
            try:
                instance = cls()
                behavior = getattr(instance, method_name)()
                behaviors.add(behavior)
            except AttributeError:
                continue  # Skip if method not implemented
        # Score based on the number of unique behaviors
        polymorphism_score = len(behaviors) * 10  # We Assign a higher score for greater diversity
        return max(polymorphism_score, 1)  # Ensure a minimal score for the attempt

    def evaluate_method_overriding(self, subclass, superclass, method_name):
        subclass_method = getattr(subclass, method_name, None)
        superclass_method = getattr(superclass, method_name, None)
        # Score for successful overriding
        return 10 if subclass_method and superclass_method and subclass_method != superclass_method else 1

    def evaluate_encapsulation(self, cls_reference):
        encapsulation_score = 0
        attributes = [attr for attr in dir(cls_reference) if attr.startswith('_') and not attr.startswith('__')]
        for attr in attributes:
            getter = f'get{attr[1:].capitalize()}'
            setter = f'set{attr[1:].capitalize()}'
            if hasattr(cls_reference, getter) and hasattr(cls_reference, setter):
                encapsulation_score += 5  # Encourage encapsulation practices
        return max(encapsulation_score, 1)  # Minimum score for attempt

    def evaluate_inheritance(self, cls_reference):
        def inheritance_depth(cls):
            if not cls:
                return 0
            if cls.__bases__:
                return 1 + max(inheritance_depth(base) for base in cls.__bases__)
            return 1

        depth = inheritance_depth(cls_reference)
        # A minimal positive score for attempting inheritance
        return max(depth, 0.1)

    def evaluate_faults(self, test_case, weights=None):
        fault_score = 0 # this accumulates penalties based on various conditions found within the test case.
        # Scale factor to moderate the impact of faults on the overall fitness score
        # This allows the balance between encouraging innovative solutions and penalizing faults to be finely tuned.
        scale_factor = 0.5
        # Standard penalties
        if test_case.get('is_negative', False) and not test_case.get('error_handled', False):
            fault_score += 20  # Increased penalty
        if test_case.get('test_type') == 'polymorphism' and test_case.get('num_polymorphic_methods', 0) < 3:
            fault_score += 15  # Increased penalty
        if test_case.get('test_type') == 'inheritance' and test_case.get('inheritance_depth', 0) < 2:
            fault_score += 12  # Increased penalty
        # Additional fault evaluations
        if test_case.get('test_type') == 'encapsulation' and test_case.get('encapsulation_score', 10) < 5:
            # Assuming a lower score indicates poor encapsulation practices
            fault_score += 18  # Increased penalty
        if test_case.get('execution_time', 0) > 50:  # Execution time in milliseconds
            # Penalize for long execution times without justification
            fault_score += 8  # Increased penalty
        if not test_case.get('has_edge_case_tests', False):
            # Checks if edge cases are tested, penalizes if not
            fault_score += 7  # Increased penalty
        if test_case.get('relies_on_happy_path', False):
            # Penalizes over-reliance on scenarios where everything goes right
            fault_score += 10  # Increased penalty
        if test_case.get('input_validation', True) == False:
            # Penalizes lack of input parameter validation
            fault_score += 15  # Increased penalty
        if not test_case.get('is_well_documented', False):
            # Checks if the test case is well-documented, penalizes if not
            fault_score += 6  # Increased penalty

        # Apply scale factor to the total fault score;
# to ensure fault penalties do not disproportionately affect the fitness score unless such weighting is desired.
        scaled_fault_score = fault_score * scale_factor

        # Consider additional weights for fault detection, if provided
        if weights:
            scaled_fault_score *= weights.get('fault_detection', 1)  # Use provided weight or default to 1

        return scaled_fault_score

    def levy_flight(self):
        sigma = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) / (
                    gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=self.nest_size)
        v = np.random.normal(0, 1, size=self.nest_size)
        step = u / abs(v) ** (1 / self.beta)
        return step

    def update_nests(self):
        for nest in self.nests:
            step_size = self.levy_flight()
            new_nest = nest + step_size * np.random.rand(*nest.shape)
            new_fitness = self.get_fitness(new_nest)
            if new_fitness < self.get_fitness(nest):
                nest[:] = new_nest
        # abandon worst nests or worst solutions
        for nest in self.nests:
            if np.random.rand() < self.pa:
                nest[:] = np.random.rand(*nest.shape)

    def find_best_solution(self):
        for nest in self.nests:
            fitness = self.get_fitness(nest)
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best_nest = nest
        return self.best_nest, self.best_fitness


# Define Simulated Annealing Algorithm Components
class SimulatedAnnealing:
    def __init__(self, initial_temperature, cooling_rate):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def initial_solution(self, cuckoo_search):
        # Use the best solution from Cuckoo Search as the initial solution for Simulated Annealing
        self.current_solution, _ = cuckoo_search.find_best_solution()
        return self.current_solution

    def get_neighbour(self, solution):
        neighbour = solution.copy()
        tweak_index = random.randint(0, len(neighbour) - 1)

        # Instead of doubling the value, let's make a smaller change
        change = random.uniform(-0.1, 0.1) * neighbour[tweak_index]
        neighbour[tweak_index] += change

        return neighbour

    def acceptance_probability(self, old_cost, new_cost):
        # Calculate the acceptance probability
        if new_cost < old_cost:
            return 1.0
        else:
            return math.exp((old_cost - new_cost) / self.temperature)

    def anneal(self, fitness_function):
        current_solution = self.current_solution
        current_cost = fitness_function(current_solution)

        while self.temperature > 1:
            new_solution = self.get_neighbour(current_solution)
            new_cost = fitness_function(new_solution)

            if self.acceptance_probability(current_cost, new_cost) > random.random():
                current_solution = new_solution
                current_cost = new_cost

            self.temperature *= 1 - self.cooling_rate

        return current_solution


# Hybrid Algorithm Integration
class HybridAlgorithm:
    def __init__(self, user_input_code):
        self.cuckoo = CuckooSearch(population_size=20, nest_size=5, pa=0.5, beta=1.0)
        self.annealing = SimulatedAnnealing(initial_temperature=10100, cooling_rate=0.03)
        self.user_input_code = user_input_code

    def generate_test_cases(self, program):
        analysis = ProgramAnalysis(program)
        analysis.extract_structure()

        return analysis.identify_test_scenarios()

    def hybrid_optimization(self, test_cases):
        # Optimize test cases using Cuckoo Search
        self.cuckoo.generate_initial_population()
        best_fitness_per_generation = []
        last_improvement_generation = 0
        best_ever_fitness = float('inf')
        NUM_GENERATIONS = 50  # Define the number of generations
        CONVERGENCE_THRESHOLD = 10  # Define convergence threshold

        for generation in range(NUM_GENERATIONS):
            self.cuckoo.update_nests()
            _, current_best_fitness = self.cuckoo.find_best_solution()

            # Update the best fitness per generation
            best_fitness_per_generation.append(current_best_fitness)

            # Update best ever fitness and last improvement generation
            if current_best_fitness < best_ever_fitness:
                best_ever_fitness = current_best_fitness
                last_improvement_generation = generation

            # Check for convergence
            if generation - last_improvement_generation > CONVERGENCE_THRESHOLD:
                break

        # Further optimization with Simulated Annealing
        self.annealing.initial_solution(self.cuckoo.best_nest)
        optimized_solution = self.annealing.anneal(self.cuckoo.get_fitness)

        convergence_generation = last_improvement_generation
        return optimized_solution, best_fitness_per_generation, convergence_generation

    def evaluate_test_cases(self, test_cases):
        optimized_solution, fitness_data, convergence_gen = self.hybrid_optimization(test_cases)
        score = self.cuckoo.get_fitness(optimized_solution)
        return score, optimized_solution, fitness_data, convergence_gen


class ProgramAnalysis:
    def __init__(self, program_code):
        self.program_code = program_code
        self.class_inheritance = {}  # Store inheritance relationships
        self.class_methods = {}  # Store methods and their parameters for each class
        self.coverage = set()  # To track covered methods

    def extract_structure(self):
        try:
            tree = ast.parse(self.program_code)
            self._parse_ast(tree)
        except SyntaxError as e:
            raise ValueError(f"Error parsing the Python code: {e}")

    def _parse_ast(self, node):
        for item in ast.walk(node):
            if isinstance(item, ast.ClassDef):
                self._process_class_definition(item)

    def _process_class_definition(self, class_node):
        class_name = class_node.name
        self.class_inheritance[class_name] = [base.id for base in class_node.bases if isinstance(base, ast.Name)]
        self.class_methods[class_name] = {method.name: self._parse_method_parameters(method)
                                          for method in class_node.body if isinstance(method, ast.FunctionDef)}

    def _parse_method_parameters(self, method_node):
        return {arg.arg: self._get_default_value(method_node, arg) for arg in method_node.args.args}

    def _get_default_value(self, function_node, arg):
        defaults_index = len(function_node.args.args) - len(function_node.args.defaults)
        arg_index = function_node.args.args.index(arg)
        if arg_index >= defaults_index:
            return repr(function_node.args.defaults[arg_index - defaults_index])
        return None

    def identify_test_inputs(self, class_name, method_name):
        method_info = self.class_methods.get(class_name, {}).get(method_name, {})
        positive_test_inputs = {}
        negative_test_inputs = {}

        for param, default in method_info.items():
            if default is not None:
                positive_test_inputs[param] = "put in a valid test input"
                negative_test_inputs[param] = self.generate_negative_input(default)
            else:
                # Placeholder logic for generating test inputs
                positive_test_inputs[param] = "put in a valid test input"
                negative_test_inputs[param] = "invalid_test_value"

        return positive_test_inputs, negative_test_inputs

    def generate_negative_input(self, default_value):
        # Improved logic for negative input generation based on type
        if isinstance(default_value, int):
            # Use a negative value for an integer, ensuring it's not just the negation (e.g., -1 for 1).
            return default_value - 1 if default_value > 0 else default_value - 2
        elif isinstance(default_value, str):
            # Return an empty string for non-empty defaults or a non-empty string for empty defaults.
            return "" if default_value else "unexpected string"
        elif isinstance(default_value, bool):
            # Flip the boolean value.
            return not default_value
        elif isinstance(default_value, float):
            # Use a negative or significantly altered value for a float.
            return -default_value if default_value > 0 else default_value - 1.1
        elif isinstance(default_value, list):
            # Return an empty list for non-empty defaults or a list with unexpected elements for empty defaults.
            return [] if default_value else ["unexpected element"]
        elif isinstance(default_value, dict):
            # Return an empty dict for non-empty defaults or a dict with unexpected key-value pairs for empty defaults.
            return {} if default_value else {"unexpected key": "unexpected value"}
        elif isinstance(default_value, tuple):
            # Return an empty tuple for non-empty defaults or a tuple with unexpected elements for empty defaults.
            return () if default_value else ("unexpected element",)
        return "invalid_test_value"

    def identify_test_scenarios(self):
        test_cases = []
        for class_name, methods in self.class_methods.items():
            # Enhanced logic to include critical methods and exclude display methods
            critical_methods = [method for method in methods if not method.startswith('display_')]
            for method in critical_methods:
                test_cases.append(self.generate_test_case_for_method(class_name, method))
                self.mark_covered(class_name, method)  # Mark critical methods as covered
        return test_cases

    def generate_test_case_for_method(self, class_name, method):
        inputs, _ = self.identify_test_inputs(class_name, method)
        expected_output = self.define_expected_outputs(class_name, method)

        return {
            'test_type': 'functional',
            'class_name': class_name,
            'method_name': method,
            'inputs': inputs,
            'expected_output': expected_output,
        }

    def define_expected_outputs(self, class_name, method_name):
        # Enhanced output definitions with specific checks for certain method patterns
        if method_name.startswith("is") or method_name.startswith("has"):
            return True  # Expect boolean true for methods starting with 'is' or 'has'
        if method_name.startswith("get"):
            return "Expected value of the attribute"
        if method_name.startswith("set"):
            return None
        if method_name.startswith("create") or method_name.startswith("add"):
            return "Success indicator or created/added object"
        if method_name.startswith("delete") or method_name.startswith("remove"):
            return "Success indicator or removed object"
        if method_name.startswith("update"):
            return "Success indicator or updated object"
        if method_name.startswith("check") or method_name.startswith("validate"):
            return True  # or False, depending on the specific validation logic
        if method_name.startswith("calculate") or method_name.startswith("compute"):
            return "Expected numeric result of the calculation"
        if method_name.startswith("find") or method_name.startswith("search"):
            return "Found object or collection of objects"
        if method_name.startswith("can") or method_name.startswith("should"):
            return True  # Assuming a positive scenario; could also be False

        return "Specific expected output based on method functionality"

    def mark_covered(self, class_name, method_name):
        self.coverage.add((class_name, method_name))

    def generate_coverage_report(self):
        total_methods = sum(len(methods) for methods in self.class_methods.values())
        covered_methods = len(self.coverage)
        uncovered_methods = total_methods - covered_methods
        coverage_percentage = (covered_methods / total_methods * 100) if total_methods else 0
        return {
            "coverage_percentage": coverage_percentage,
            "covered_methods": covered_methods,
            "uncovered_methods": uncovered_methods
        }


def collect_metrics(algorithm_name, execution_time, best_fitness, coverage_percentage, mode='w',
                    filename='new_hybrid_algorithm_metrics.txt'):
    metrics_content = (
        f"Algorithm: {algorithm_name}\n"
        f"Execution Time: {execution_time} \n"
        f"Best Fitness: {best_fitness}\n"
        f"Coverage: {coverage_percentage}%\n\n"
    )

    with open(filename, mode) as f:
        f.write(metrics_content)


# Main Function to Run the Tool
def main():
    # Example usage of the tool
    program_code = """

import random
import math
from scipy.special import gamma
import numpy as np
import ast
import time
import astor
# Define Cuckoo Search Algorithm Components


class CuckooSearch:
    def __init__(self, population_size, nest_size, pa, beta=3.0):
        
        self.population_size = population_size
        self.nest_size = nest_size
        self.pa = pa
        self.beta = beta
        self.nests = np.random.rand(self.population_size, self.nest_size)
        self.best_nest = None
        self.best_fitness = float('inf')
        self.previous_best_fitness = float('inf')  # Initialize previous_best_fitness


    def generate_initial_population(self):
        
        self.nests = np.random.rand(self.population_size, self.nest_size)

    def get_fitness(self, nest):
        
        return self.fitness(nest)

    def normalize_and_weight(self, score, max_score, weight, transform_type='linear'):
        normalized_score = min(score / max_score, 1.0)
        weighted_score = weight * normalized_score

        if transform_type == 'logarithmic':
            # Apply logarithmic transformation, ensuring score is always > 0
            weighted_score = np.log(weighted_score + 1)
        elif transform_type == 'power':
            # Apply power transformation with a power of 2 for demonstration
            weighted_score = weighted_score ** 2
        elif transform_type == 'exponential':
            # Apply exponential decay transformation with adjustment
            weighted_score = np.exp(-weighted_score) * weight

        return weighted_score

    def fitness(self, test_case):
        score = 0
        weights = {
            'inheritance': 2,
            'polymorphism': 2,
            'encapsulation': 1.5,
            'method_override': 1.5,
            'performance': 2.5,
            'error_handling': 3
        }
        max_scores = {
            'inheritance': 5,
            'polymorphism': 10,
            'encapsulation': 5,
            'method_override': 5,
            'performance': 100,
            'error_handling': 10
        }
        transform_types = {
            'inheritance': 'logarithmic',
            'polymorphism': 'power',
            'encapsulation': 'exponential',
            'method_override': 'logarithmic',
            'performance': 'power',
            'error_handling': 'exponential'
        }

        for test_type, weight in weights.items():
            test_score = test_case.get(f'{test_type}_score', 0)
            transform_type = transform_types[test_type]
            score += self.normalize_and_weight(test_score, max_scores[test_type], weight, transform_type)

        fault_score = self.evaluate_faults(test_case)
        # Consider using a non-linear transformation for the fault score as well
        fault_score_transformed = np.exp(-fault_score)
        score += fault_score_transformed

        # Final transformation to ensure non-linear differentiation
        score = 1 / (1 + score) if score != 0 else float('inf')
        # Further penalize with an exponential function
        score = np.exp(-10 * score)

        return score


    def evaluate_faults(self, test_case, weights=None):
        
        fault_score = 0

        # Safe retrieval of values, defaulting to 0 if not found or if None
        def safe_get(key, default=0):
            value = test_case.get(key, default)
            return default if value is None else value

        # Example penalty for not handling errors in negative tests
        if safe_get('is_negative', False) and not safe_get('error_handled', False):
            fault_score += 10

        # Polymorphism fault: Insufficient testing of polymorphic methods
        if test_case.get('test_type') == 'polymorphism' and test_case.get('num_polymorphic_methods', 0) < 3:
            fault_score += 10

        # Inheritance fault: Shallow inheritance testing
        if test_case.get('test_type') == 'inheritance' and test_case.get('inheritance_depth', 0) < 2:
            fault_score += 8

        # Encapsulation fault: Poor validation of encapsulation
        encapsulation_score = test_case.get('encapsulation_validation_score', 0)
        if test_case.get('test_type') == 'encapsulation' and encapsulation_score < 0.5:
            # Assuming a score below 0.5 indicates poor encapsulation validation
            fault_score += 12

        # Performance fault: Unjustifiably high execution time
        execution_time = test_case.get('execution_time', 100)
        if test_case.get('test_type') == 'performance' and execution_time > 50:
            # Assuming performance tests should not exceed 50ms execution time under normal conditions
            fault_score += 6

        # Error handling fault: Lack of error handling in negative test cases
        if test_case.get('is_negative', False) and not test_case.get('error_handled', False):
            fault_score += 14

        # Consider weights for fault detection, if weights are provided
        if weights:
            fault_score *= weights.get('fault_detection', 1)  # Default weight is 1 if not specified

        # Concurrency issues penalty
        concurrency_issues_tested = test_case.get('concurrency_issues_tested', 0)
        if concurrency_issues_tested < 2:  # Assuming less than 2 tests for concurrency issues indicate insufficient testing
            fault_score += 12

        # Resource utilization penalty
        resource_utilization_score = test_case.get('resource_utilization_score', 0)
        if resource_utilization_score < 5:  # Assuming scores below 5 indicate poor resource utilization testing
            fault_score += 6

        # Security vulnerability checks penalty
        security_checks = test_case.get('security_checks', 0)
        if security_checks < 2:  # Assuming less than 2 security checks indicates insufficient security testing
            fault_score += 14

        # Complexity penalty: Increase fault score for low complexity or coverage
        complexity_score = test_case.get('complexity_score', 0)
        if complexity_score < 3:  # Assuming a complexity score below 3 indicates low complexity or coverage
            fault_score += 10

        # Code style and best practices penalty
        code_style_score = test_case.get('code_style_score', 0)
        if code_style_score < 5:  # Assuming scores below 5 indicate poor adherence to code style or best practices
            fault_score += 8

        return fault_score

    def levy_flight(self):
        
        sigma = (gamma(1 + self.beta) * np.sin(np.pi * self.beta / 2) /
                 (gamma((1 + self.beta) / 2) * self.beta * 2 ** ((self.beta - 1) / 2))) ** (1 / self.beta)
        u = np.random.normal(0, sigma, size=self.nest_size)
        v = np.random.normal(0, 1, size=self.nest_size)
        step = u / np.abs(v) ** (1 / self.beta)

        return step

    def update_nests(self, generation):
        improvement_threshold = 0.01  # Example threshold
        if generation > 0 and (self.best_fitness - self.previous_best_fitness) / self.previous_best_fitness < improvement_threshold:
            self.pa *= 1.1  # Increase pa by 10%
        for i, nest in enumerate(self.nests):
            step_size = self.levy_flight()
            new_nest = nest + step_size * np.random.rand(*nest.shape)
            new_fitness = self.get_fitness(new_nest)
            if new_fitness < self.get_fitness(nest):
                self.nests[i] = new_nest
                if new_fitness < self.best_fitness:
                    self.best_nest = new_nest
                    self.best_fitness = new_fitness
            elif np.random.rand() < self.pa:
                self.nests[i] = np.random.rand(*self.nests[i].shape)
        self.previous_best_fitness = self.best_fitness

        self.abandon_worse_nests()


    def abandon_worse_nests(self):
        for i, _ in enumerate(self.nests):
            if np.random.rand() < self.pa:
                self.nests[i] = np.random.rand(*self.nests[i].shape)

    def find_best_solution(self):
        # The best solution is already found during the nest updates
        return self.best_nest, self.best_fitness
# Define Simulated Annealing Algorithm Components
class SimulatedAnnealing:
    def __init__(self, initial_temperature, cooling_rate):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def initial_solution(self, cuckoo_search):
        # Use the best solution from Cuckoo Search as the initial solution for Simulated Annealing
        self.current_solution, _ = cuckoo_search.find_best_solution()
        return self.current_solution

    def get_neighbour(self, solution):
        neighbour = solution.copy()
        tweak_index = random.randint(0, len(neighbour) - 1)

        # Instead of doubling the value, let's make a smaller change
        change = random.uniform(-0.1, 0.1) * neighbour[tweak_index]
        neighbour[tweak_index] += change

        return neighbour

    def acceptance_probability(self, old_cost, new_cost):
        # Calculate the acceptance probability
        if new_cost < old_cost:
            return 1.0
        else:
            return math.exp((old_cost - new_cost) / self.temperature)

    def anneal(self, fitness_function):
        successful_attempts = 0
        attempts_since_last_success = 0
        while self.temperature > 1:
            new_solution = self.get_neighbour(self.current_solution)
            new_cost = fitness_function(new_solution)
            if self.acceptance_probability(self.current_cost, new_cost) > random.random():
                self.current_solution = new_solution
                self.current_cost = new_cost
                successful_attempts += 1
                attempts_since_last_success = 0
            else:
                attempts_since_last_success += 1
            if attempts_since_last_success > 10:  # If no success in the last 10 attempts, increase cooling rate
                self.cooling_rate *= 1.05
            elif successful_attempts % 10 == 0:  # Every 10 successful attempts, decrease the cooling rate
                self.cooling_rate /= 1.05
            self.temperature *= 1 - self.cooling_rate
        return self.current_solution

# Hybrid Algorithm Integration
class HybridAlgorithm:
    def __init__(self, user_input_code):
        self.cuckoo = CuckooSearch(population_size=14, nest_size=20, pa=1.0, beta=3.0)
        self.annealing = SimulatedAnnealing(initial_temperature=10010, cooling_rate=0.0015)
        self.user_input_code = user_input_code

    def generate_test_cases(self, program):
        analysis = ProgramAnalysis(program)
        analysis.extract_structure()

        return analysis.identify_test_scenarios()

    def hybrid_optimization(self, test_cases):
        # Optimize test cases using Cuckoo Search
        self.cuckoo.generate_initial_population()
        best_fitness_per_generation = []
        last_improvement_generation = 0
        best_ever_fitness = float('inf')
        NUM_GENERATIONS = 50  # Define the number of generations
        CONVERGENCE_THRESHOLD = 10  # Define convergence threshold

        for generation in range(NUM_GENERATIONS):
            self.cuckoo.update_nests(generation)
            _, current_best_fitness = self.cuckoo.find_best_solution()

            # Update the best fitness per generation
            best_fitness_per_generation.append(current_best_fitness)

            # Update best ever fitness and last improvement generation
            if current_best_fitness < best_ever_fitness:
                best_ever_fitness = current_best_fitness
                last_improvement_generation = generation

            # Check for convergence
            if generation - last_improvement_generation > CONVERGENCE_THRESHOLD:
                break

        # Further optimization with Simulated Annealing
        self.annealing.initial_solution(self.cuckoo.best_nest)
        optimized_solution = self.annealing.anneal(self.cuckoo.get_fitness)

        convergence_generation = last_improvement_generation
        return optimized_solution, best_fitness_per_generation, convergence_generation

    def evaluate_test_cases(self, test_cases):
        optimized_solution, fitness_data, convergence_gen = self.hybrid_optimization(test_cases)
        score = self.cuckoo.get_fitness(optimized_solution)
        return score, optimized_solution, fitness_data, convergence_gen






class ProgramAnalysis:
    def __init__(self, program_code):
        # Correct the code first
        self.program_code = self.correct_multiline_strings(program_code)
        self.class_inheritance = {}
        self.class_methods = {}
        self.coverage = set()

    def correct_multiline_strings(self, code):
        
        tree = ast.parse(code)
        self.visit(tree)
        corrected_code = astor.to_source(tree)
        return corrected_code

    def visit(self, node):
        
        for field, value in ast.iter_fields(node):
            if isinstance(value, str):
                setattr(node, field, self.correct_multiline_string(value))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.Str):
                        value[i] = ast.Str(s=self.correct_multiline_string(item.s))
                    elif isinstance(item, ast.JoinedStr):
                        self.correct_f_string(item)
                    else:
                        self.visit(item)
            elif isinstance(value, ast.AST):
                self.visit(value)

    @staticmethod
    def correct_multiline_string(string):

        return string.replace('', ' ')

    def correct_f_string(self, node):
        
        for part in node.values:
            if isinstance(part, ast.Str):
                part.s = self.correct_multiline_string(part.s)
            else:
                self.visit(part)


    def extract_structure(self):
        try:
            tree = ast.parse(self.program_code)
            self._parse_ast(tree)
        except SyntaxError as e:
            raise ValueError(f"Error parsing the Python code: {e}")

    def _parse_ast(self, node):
        for item in ast.walk(node):
            if isinstance(item, ast.ClassDef):
                self._process_class_definition(item)

    def _process_class_definition(self, class_node):
        class_name = class_node.name
        self.class_inheritance[class_name] = [base.id for base in class_node.bases if isinstance(base, ast.Name)]
        self.class_methods[class_name] = {method.name: self._parse_method_parameters(method)
                                          for method in class_node.body if isinstance(method, ast.FunctionDef)}
        # Detect and process method overrides
        self._detect_method_overrides(class_node)

    def _detect_method_overrides(self, class_node):
        class_name = class_node.name
        for base in class_node.bases:
            base_name = base.id if isinstance(base, ast.Name) else None
            if base_name and base_name in self.class_methods:
                self._compare_methods_for_override(class_name, base_name)

    def _compare_methods_for_override(self, class_name, base_name):
        class_methods = self.class_methods[class_name]
        base_methods = self.class_methods[base_name]
        for method in class_methods:
            if method in base_methods:
                print(f"Method {method} in class {class_name} overrides method from {base_name}")

    def _parse_method_parameters(self, method_node):
        return {arg.arg: self._get_default_value(method_node, arg) for arg in method_node.args.args}

    def _get_default_value(self, function_node, arg):
        defaults_index = len(function_node.args.args) - len(function_node.args.defaults)
        arg_index = function_node.args.args.index(arg)
        if arg_index >= defaults_index:
            return repr(function_node.args.defaults[arg_index - defaults_index])
        return None

    def identify_test_inputs(self, class_name, method_name):
        method_info = self.class_methods.get(class_name, {}).get(method_name, {})
        positive_test_inputs = {}
        negative_test_inputs = {}

        for param, default in method_info.items():
            if default is not None:
                positive_test_inputs[param] = default  # Use the default value
                negative_test_inputs[param] = self.generate_negative_input(default)
            else:
                # Placeholder logic for generating test inputs
                positive_test_inputs[param] = "valid_test_value"
                negative_test_inputs[param] = "invalid_test_value"

        return positive_test_inputs, negative_test_inputs

    def generate_negative_input(self, default_value):
        # Improved logic for negative input generation based on type
        if isinstance(default_value, int):
            return -default_value  # Example: Use negative value for int
        elif isinstance(default_value, str) and default_value:
            return ""  # Example: Empty string for non-empty default
        # Extend logic for other types as necessary
        return "invalid_test_value"

    def identify_test_scenarios(self):
        test_cases = []
        for class_name, methods in self.class_methods.items():
            # Enhanced logic to include critical methods and exclude display methods
            critical_methods = [method for method in methods if not method.startswith('display_')]
            for method in critical_methods:
                test_cases.append(self.generate_test_case_for_method(class_name, method))
                self.mark_covered(class_name, method)  # Mark critical methods as covered
        return test_cases

    def generate_test_case_for_method(self, class_name, method):
        inputs, _ = self.identify_test_inputs(class_name, method)
        expected_output = self.define_expected_outputs(class_name, method)
        # Adding mock scores for fitness evaluation
        test_scores = {
            'inheritance_score': random.randint(1, 5),
            'polymorphism_score': random.randint(1, 10),
            'encapsulation_score': random.randint(1, 5),
            'method_override_score': random.randint(1, 5),
            'performance_score': random.randint(1, 100),
            'error_handling_score': random.randint(1, 10)
        }
        return {
            'test_type': 'functional',
            'class_name': class_name,
            'method_name': method,
            'inputs': inputs,
            'expected_output': expected_output,
            **test_scores  # Merge test scores into the test case dictionary
        }

    def define_expected_outputs(self, class_name, method_name):
        # Enhanced output definitions with specific checks for certain method patterns
        if method_name.startswith("is") or method_name.startswith("has"):
            return True  # Expect boolean true for methods starting with 'is' or 'has'
        # Extend with more heuristics as needed
        return "Specific expected output based on method functionality"

    def mark_covered(self, class_name, method_name):
        self.coverage.add((class_name, method_name))

    def generate_coverage_report(self):
        # Assuming all methods in self.class_methods should be covered
        all_methods = set()
        for class_name, methods in self.class_methods.items():
            for method in methods:
                all_methods.add((class_name, method))

        covered_methods = self.coverage
        uncovered_methods = all_methods - covered_methods

        coverage_percentage = (len(covered_methods) / len(all_methods) * 100) if all_methods else 0

        # Convert the sets of tuples into a list of method names
        covered_methods_list = [f'{class_name}.{method}' for class_name, method in covered_methods]
        uncovered_methods_list = [f'{class_name}.{method}' for class_name, method in uncovered_methods]

        return {
            "coverage_percentage": coverage_percentage,
            "covered_methods": covered_methods_list,  # This should be a list
            "uncovered_methods": uncovered_methods_list,  # This should also be a list
        }

    def collect_metrics(algorithm_name, execution_time, best_fitness, coverage_percentage, mode='w',
                        filename='new_hybrid_algorithm_metrics.txt'):
        metrics_content = (
            f"Algorithm: {algorithm_name}",
            f"Execution Time: {execution_time}",
            f"Best Fitness: {best_fitness}",
            f"Coverage: {coverage_percentage}"
        )
    
        with open(filename, mode) as f:
            f.write(metrics_content)


    # Main Function to Run the Tool
    def main():
        # Example usage of the tool
        program_code = Test_code
    
        start_time = time.perf_counter()
        analysis = ProgramAnalysis(program_code)
        analysis.extract_structure()
        test_scenarios = analysis.identify_test_scenarios()
        hybrid_tool = HybridAlgorithm(user_input_code=program_code)
    
        # Generate test cases
        test_cases = hybrid_tool.generate_test_cases(program_code)
    
        best_fitness = float('inf')  # Initialize best_fitness with the highest possible value
        best_test_case = None  # To keep track of the test case with the best fitness
    
        print("Generated Test Cases and Fitness Scores:")
        for test_case in test_cases:
            fitness_score = hybrid_tool.cuckoo.get_fitness(test_case)
            if fitness_score < best_fitness:
                best_fitness = fitness_score
                best_test_case = test_case
            print(f"Test Case: {test_case}, Fitness Score: {fitness_score}")
    
        print(f"Best fitness is {best_fitness}")  # Move this print statement outside the loop
    
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"The time taken to generate test cases was {execution_time}")
        # Generate and print the coverage report
        coverage_report = analysis.generate_coverage_report()
        print("Coverage Report:")
        print(f"Coverage Percentage: {coverage_report['coverage_percentage']:.2f}%")
        print("Covered Methods:", coverage_report['covered_methods'])
        print("Uncovered Methods:", coverage_report['uncovered_methods'])
    
        collect_metrics("HybridAlgorithm", execution_time, best_fitness, coverage_report['coverage_percentage'])
    
    
    if __name__ == "__main__":
        main()


"""

    start_time = time.perf_counter()
    analysis = ProgramAnalysis(program_code)
    analysis.extract_structure()
    analysis.identify_test_scenarios()
    hybrid_tool = HybridAlgorithm(user_input_code=program_code)
    test_cases = hybrid_tool.generate_test_cases(program_code) # Generate test cases
    best_fitness_score = float('inf')
    best_test_case = None
    for test_case in test_cases:
        current_fitness_score = hybrid_tool.cuckoo.get_fitness(test_case)
        print(f"Test Case: {test_case},\n Fitness Score: {current_fitness_score}")
        if current_fitness_score < best_fitness_score:
            best_fitness_score = current_fitness_score
            best_test_case = test_case
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Best Fitness Score: {best_fitness_score}")
    print(f"Best Test Case: {best_test_case}")
    print(f"The time taken to generate test cases was {execution_time}")
    # Generate and print the coverage report
    coverage_report = analysis.generate_coverage_report()
    print("\nCoverage Report:")
    print(f"Coverage Percentage: {coverage_report['coverage_percentage']:.2f}%")
    print("Covered Methods:", coverage_report['covered_methods'])
    print("Uncovered Methods:", coverage_report['uncovered_methods'])

    collect_metrics("HybridAlgorithm", execution_time, best_fitness_score, coverage_report['coverage_percentage'])


if __name__ == "__main__":
    main()
