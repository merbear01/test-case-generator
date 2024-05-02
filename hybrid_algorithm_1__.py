import random
import math
from scipy.special import gamma
import numpy as np
import ast
import time
# from memory_profiler import profile
import logging
import cProfile
import pstats
# Define Cuckoo Search Algorithm Components



# Ensure logging captures all I/O simulation
logging.basicConfig(filename='algorithm_efficiency.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

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
        min_fitness_score = 0.001  # minimum threshold for the fitness score is set here
        base_score = 0.5
        # Check for missing keys and assign default values if necessary
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
        random_variation = random.uniform(-0.5, 0.5)  # Adjust the range as needed to create impact of randomness
        evaluation_score += random_variation
        # Ensure a minimum fitness score even in the presence of faults
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
        fault_score = 0
        # Scale factor to moderate the impact of faults on the overall fitness score
        scale_factor = 0.5  # Adjust this value based on desired impact
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

        # Apply scale factor to the total fault score
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
    program_code = """Program code"""

    start_time = time.perf_counter()
    analysis = ProgramAnalysis(program_code)
    analysis.extract_structure()
    analysis.identify_test_scenarios()
    hybrid_tool = HybridAlgorithm(user_input_code=program_code)
    test_cases = hybrid_tool.generate_test_cases(program_code)  # Generate test cases
    best_fitness_score = float('inf')
    best_test_case = None
    for test_case in test_cases:
        current_fitness_score = hybrid_tool.cuckoo.get_fitness(test_case)
        logging.info(f"Evaluating test case: {test_case}, Fitness Score: {current_fitness_score}")
        if current_fitness_score < best_fitness_score:
            best_fitness_score = current_fitness_score
            best_test_case = test_case
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Best Fitness Score: {best_fitness_score}")
    print(f"Best Test Case: {best_test_case}")
    logging.info(f"Total Execution Time: {execution_time} seconds")
    # Generate and print the coverage report
    coverage_report = analysis.generate_coverage_report()
    print("\nCoverage Report:")
    print(f"Coverage Percentage: {coverage_report['coverage_percentage']:.2f}%")
    print("Covered Methods:", coverage_report['covered_methods'])
    print("Uncovered Methods:", coverage_report['uncovered_methods'])
    logging.info(f"Coverage Report: {coverage_report}")


    collect_metrics("HybridAlgorithm", execution_time, best_fitness_score, coverage_report['coverage_percentage'])
def profile_main():
    profiler = cProfile.Profile()
    profiler.enable()
    main()
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats()

if __name__ == "__main__":
    profile_main()


