"""
Teaching learning based optimization algorithm for Test Case Generation

This code implements a Teaching learning based optimization algorithm to generate test cases for program analysis.

References:
- andaviaco . (2017). tblo . Github. https://github.com/andaviaco/tblo/blob/master/src/tblo.py
- dothihai . (2019). TLBO . Github . https://github.com/dothihai/TLBO/blob/master/tblo.py
- Jifeng Wu. (2023). static-import-analysis-for-pure-python-projects. Github. https://github.com/abbaswu/static-import-analysis-for-pure-python-projects/tree/main/static_import_analysis
- Boadzie, D. (2023) Introduction to Abstract Syntax Trees in Python, Earthly Blog. Available at: https://earthly.dev/blog/python-ast/.

"""


import numpy as np
import random
import ast
import time

class TLBOAlgorithm:
    def __init__(self, population_size, iterations, program_analysis):
        self.population_size = population_size  # This should be an integer
        self.iterations = iterations
        self.program_analysis = program_analysis
        self.population = [self.program_analysis.generate_test_case() for _ in range(self.population_size)]

    def fitness(self, test_case):
        min_fitness_score = 0.001  # Set a minimum threshold for the fitness score

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

        # Adjust the range and impact of randomness
        random_variation = random.uniform(-0.5, 0.5)  # Adjust the range as needed
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
        polymorphism_score = len(behaviors) * 10  # Assigning a higher score for greater diversity
        return max(polymorphism_score, 1)  # Ensure a minimal score for the attempt

    def evaluate_method_overriding(self, subclass, superclass, method_name):
        # Ensure method_name is a string
        if not isinstance(method_name, str):
            method_name = str(method_name)

        # Check if subclass and superclass are class objects and not None
        if subclass and superclass and isinstance(subclass, type) and isinstance(superclass, type):
            subclass_method = getattr(subclass, method_name, None)
            superclass_method = getattr(superclass, method_name, None)
            # Score for successful overriding
            return 10 if subclass_method and superclass_method and subclass_method != superclass_method else 1
        else:
            # Handle the case where subclass or superclass are not class objects
            return 0  # Assuming 0 as a score for non-valid class object comparison

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
        """
        Evaluate test cases for potential faults and assign a fault score. A higher fault score indicates
        more or severe faults detected in the test case.

        Faults evaluated can include but are not limited to:
        - Insufficient polymorphic method testing
        - Shallow inheritance depth
        - Poor encapsulation validation
        - High execution time without justification
        - Lack of error handling in negative test cases
        - Misuse of class attributes or methods
        - Inadequate testing of edge cases
        - Over-reliance on 'happy path' testing without considering negative scenarios
        - Insufficient validation of input parameters
        - Poor test case documentation or unclear testing objectives
        """
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
    def teacher_phase(self):
        # Identify the best solution in the population as the teacher
        teacher = max(self.population, key=lambda x: self.fitness(x))
        teacher_fitness = self.fitness(teacher)

        # Improve other solutions based on the teacher
        for i in range(self.population_size):
            if self.population[i] != teacher:
                # Create a new solution by learning from the teacher
                # For simplicity, let's modify a numeric attribute of the test case
                # In a real scenario, you'd adjust attributes according to your test case structure
                new_solution = self.population[i].copy()
                if 'execution_time' in new_solution:
                    # Example of adjusting the execution time attribute based on the teacher's value
                    r = np.random.rand()  # Random number between 0 and 1
                    new_solution['execution_time'] += r * (teacher['execution_time'] - new_solution['execution_time'])

                # Evaluate the new solution
                new_fitness = self.fitness(new_solution)
                if new_fitness > self.fitness(self.population[i]):
                    # Replace the old solution with the new one if it's better
                    self.population[i] = new_solution

    def learner_phase(self):
        # Learners learn from each other
        for i in range(self.population_size):
            # Randomly select another learner
            j = i
            while j == i:
                j = np.random.randint(0, self.population_size)

            # Assume learning is beneficial and always occurs
            if self.fitness(self.population[j]) > self.fitness(self.population[i]):
                # Create a new solution by learning from another learner
                new_solution = self.population[i].copy()
                # For simplicity, adjust a numeric attribute as an example
                if 'execution_time' in new_solution:
                    r = np.random.rand()  # Random number between 0 and 1
                    new_solution['execution_time'] += r * (
                                self.population[j]['execution_time'] - new_solution['execution_time'])

                # Evaluate the new solution
                new_fitness = self.fitness(new_solution)
                if new_fitness > self.fitness(self.population[i]):
                    # Replace the old solution with the new one if it's better
                    self.population[i] = new_solution

    def run(self):
        # Main loop to run the algorithm for a set number of iterations
        for _ in range(self.iterations):
            self.teacher_phase()
            self.learner_phase()

            for test_case in self.population:
                self.simulate_test_case_execution(test_case)

            # Further logic as necessary...
        # Identify and return the best solution and its fitness from the final population
        self.best_solution = max(self.population, key=lambda x: self.fitness(x))
        self.best_fitness = self.fitness(self.best_solution)
        return self.best_solution, self.best_fitness

    def simulate_test_case_execution(self, test_case):
        # This is a simplified example; you'd implement logic based on your test cases
        class_name = test_case['class_name']
        method_name = test_case['method_name']
        self.program_analysis.mark_method_as_covered(class_name, method_name)




# Example usage:
class ProgramAnalysis:
    def __init__(self, program_code):
        self.program_code = program_code
        self.classes = {}  # Stores class names and their methods
        self.inheritance_tree = {}  # Stores inheritance relationships
        self.covered_methods = set()  # Track covered methods
        self._analyze_code()

    def _analyze_code(self):
        """
        Analyze the program code to extract classes, methods, and inheritance relationships.
        """
        tree = ast.parse(self.program_code)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                self.classes[class_name] = methods

                # Check for inheritance
                bases = [base.id for base in node.bases if isinstance(base, ast.Name)]
                self.inheritance_tree[class_name] = bases

    def generate_test_case(self):
        """
        Generate a random test case based on the program analysis.
        """
        test_case = {
            'test_type': random.choice(['inheritance', 'polymorphism', 'encapsulation', 'method_override']),
            'class_name': random.choice(list(self.classes.keys())),
            'method_name': None,
            'inheritance_depth': None,
            'num_polymorphic_methods': None,
            'encapsulation_validation_score': None,
            'override_complexity': None,
            'execution_time': random.randint(1, 100),
            'is_negative': random.choice([True, False]),
            'error_handled': None
        }

        class_name = test_case['class_name']
        methods = self.classes.get(class_name, [])

        if methods:
            test_case['method_name'] = random.choice(methods)

        if test_case['test_type'] == 'inheritance' and class_name in self.inheritance_tree:
            test_case['inheritance_depth'] = len(self.inheritance_tree[class_name])

        if test_case['test_type'] == 'polymorphism':
            test_case['num_polymorphic_methods'] = random.randint(1, len(methods))

        if test_case['test_type'] == 'encapsulation':
            test_case['encapsulation_validation_score'] = random.uniform(0, 1)

        if test_case['test_type'] == 'method_override':
            test_case['override_complexity'] = random.choice(['Low', 'Medium', 'High'])

        if test_case['is_negative']:
            test_case['error_handled'] = random.choice([True, False])

        return test_case

    def mark_method_as_covered(self, class_name, method_name):
        self.covered_methods.add((class_name, method_name))

    def generate_coverage_report(self):
        total_methods = sum(len(methods) for methods in self.classes.values())
        covered_methods = len(self.covered_methods)
        uncovered_methods = total_methods - covered_methods
        coverage_percentage = (covered_methods / total_methods * 100) if total_methods else 0
        return {
            "coverage_percentage": coverage_percentage,
            "covered_methods": covered_methods,
            "uncovered_methods": uncovered_methods
        }


def collect_metrics(algorithm_name, execution_time, best_fitness, coverage_percentage, mode='w',
                    filename='TLBOalgorithm_metrics.txt'):
    metrics_content = (
        f"Algorithm: {algorithm_name}\n"
        f"Execution Time: {execution_time} \n"
        f"Best Fitness: {best_fitness}\n"
        f"Coverage: {coverage_percentage}%\n\n"
    )

    with open(filename, mode) as f:
        f.write(metrics_content)


def main():
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

    program_analysis = ProgramAnalysis(program_code)

    # Initialize TLBOAlgorithm with the program analysis
    tlbo = TLBOAlgorithm(population_size=5, iterations=100, program_analysis=program_analysis)

    # Run the TLBO algorithm
    best_solution, best_fitness = tlbo.run()

    # Output the results
    print("Generated Test Cases:")
    for test_case in tlbo.population:
        print(test_case)


    print("\nBest Solution:", best_solution)
    print("Best Fitness:", best_fitness)
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"The time taken to generate test cases was {execution_time}")


    # After running the TLBO algorithm, generate the coverage report
    coverage_report = program_analysis.generate_coverage_report()
    print("\nCoverage Report:")
    print(f"Coverage Percentage: {coverage_report['coverage_percentage']:.2f}%")
    print(f"Covered Methods: {coverage_report['covered_methods']}")
    print(f"Uncovered Methods: {coverage_report['uncovered_methods']}")

    collect_metrics("TLBOalgorithm", execution_time, best_fitness, coverage_report['coverage_percentage'])



if __name__ == "__main__":
    main()