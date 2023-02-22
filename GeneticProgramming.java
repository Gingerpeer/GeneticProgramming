/*
 To do:
  - Concept of the genetic algorithm.
    - A genetic algorithm is a problem-solving technique that imitates the process of natural selection. 
    - It starts with a population of potential solutions to a problem, and then selects the fittest ones to "reproduce" and create a new population of solutions.
    - Each solution is represented as a string of "genes", which are evaluated based on their fitness or how well they solve the problem. 
    - The fittest solutions are selected to "mate" or combine their genes to create new solutions, which are then evaluated again for their fitness.
    - This process is repeated over multiple generations, with each generation improving upon the previous one as the fittest solutions are selected and reproduced. 
    - Eventually, the genetic algorithm converges on the best solution to the problem, which is the one with the highest fitness in the final generation. 
  - Modify your code to implement the genetic algorithm instead of a random search.
    - Representation of individuals (candidate solutions) - We can represent an individual as an expression tree. An expression tree is built using the provided classes: ConstNode,   VariableNode, BinaryOpNode, and UnaryOpNode.
    - Fitness function - We need to define a function that evaluates how well an individual solves the problem at hand. In this case, we can use the mean squared error (MSE) between the actual output of the function and the output predicted by the individual. For a given input x, the actual output can be computed using the provided function actualOutput(double x), and the predicted output can be computed using the expression tree of the individual.
    - Genetic operators - We need to define the following genetic operators:
      a. Selection: We use tournament selection. Given a population, we choose two individuals at random and select the one with the higher fitness. We repeat this process to obtain the desired number of individuals for the next generation.
      b. Crossover: We use one-point crossover. Given two parent individuals, we choose a random node in each tree and swap their subtrees to create two offspring individuals.
      c. Mutation: We use subtree mutation. Given an individual, we choose a random node in its tree and replace it with a randomly generated subtree.
    - Termination condition - We stop the algorithm after a fixed number of generations.
  - Choose the initial population of individuals by creating random expressions.
  - Evaluate the fitness of each individual in the population by comparing their output to a set of sample data.
  - Set a termination condition for the algorithm, such as a time limit or a sufficient fitness level achieved.
  - Select the best-ranking individuals in the population to reproduce.
  - Breed the new generation of individuals through crossover and/or mutation to create offspring.
  - Evaluate the individual fitnesses of the offspring.
  - Replace the lower-ranked part of the population with the offspring.
  - Keep track of the best expression found so far and output the generation number, RMS error, and expression every time a new best expression is found.
  - Compare the performance of your genetic algorithm to the random search algorithm by allowing both to run for 60 seconds and keeping track of the best RMS error and expression for each.
  - Use the old Genetic Algorithms demonstration applet to get a better understanding of how genetic algorithms work.
  - Consider using LISP programming language and tree structures to represent and manipulate the expressions as programs for genetic programming.
  - Experiment with different tuning options and decisions for the genetic algorithm to see how it affects the results.
  - Remember that this is an experimental lab and the results may vary depending on the details of your implementation.
 */
import java.util.*;

public class GeneticProgramming {
  static double actualOutput(double x) {
    return Math.sin(x);
  }
  private static ExpNode tournamentSelection(List<ExpNode> population, Map<ExpNode, Double> fitnesses) {
    int tournamentSize = 5;
    ExpNode best = null;
    double bestFitness = Double.POSITIVE_INFINITY;
    for (int i = 0; i < tournamentSize; i++) {
        int index = (int) (Math.random() * population.size());
        ExpNode individual = population.get(index);
        double fitness = fitnesses.get(individual);
        if (fitness < bestFitness) {
            bestFitness = fitness;
            best = individual;
        }
    }
    return best;
}
private static void subtreeMutation(ExpNode node, int maxHeight) {
  if (maxHeight == 0) {
      if (Math.random() < 0.5) {
          node = new ConstNode(Math.random() * 10 - 5);
      } else {
          node = new VariableNode();
      }
  } else {
      if (node instanceof BinaryOpNode) {
          BinaryOpNode opNode = (BinaryOpNode) node;
          if (Math.random() < 0.5) {
              subtreeMutation(opNode.left, maxHeight - 1);
          } else {
              subtreeMutation(opNode.right, maxHeight - 1);
          }
      }
  }
}

static void onePointCrossover(ExpNode parent1, ExpNode parent2, int maxHeight) {
  List<ExpNode> nodes1 = new ArrayList<>();
  List<ExpNode> nodes2 = new ArrayList<>();
  collectNodes(parent1, nodes1, maxHeight);
  collectNodes(parent2, nodes2, maxHeight);
  if (nodes1.size() > 1 && nodes2.size() > 1) {
      int index1 = (int) (Math.random() * (nodes1.size() - 1)) + 1;
      int index2 = (int) (Math.random() * (nodes2.size() - 1)) + 1;
      ExpNode node1 = nodes1.get(index1);
      ExpNode node2 = nodes2.get(index2);
      ExpNode parent1Copy = parent1.copy();
      node1.replaceWith(node2);
      node2.replaceWith(parent1Copy);
  }
}

static void collectNodes(ExpNode node, List<ExpNode> nodes, int maxHeight) {
  if (node instanceof BinaryOpNode) {
      BinaryOpNode opNode = (BinaryOpNode) node;
      if (opNode.left.getHeight() < maxHeight) {
          collectNodes(opNode.left, nodes, maxHeight);
      }
      if (opNode.right.getHeight() < maxHeight) {
          collectNodes(opNode.right, nodes, maxHeight);
      }
  } else if (node instanceof UnaryOpNode) {
      UnaryOpNode opNode = (UnaryOpNode) node;
      if (opNode.child.getHeight() < maxHeight) {
          collectNodes(opNode.child, nodes, maxHeight);
      }
  }
  nodes.add(node);
}


  public static void main(String[] args) {
    
    // Set up the problem
    double[] input = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0};
    // double[] output = {0.0, 0.09983341664682815, 0.19866933079506122, 0.2955202066613396, 0.38941834230865113,
    //                    0.479425538604203, 0.5646424733950354, 0.6442176872376911, 0.7173560908995228,
    //                    0.7833269096274834, 0.8414709848078965};
    int populationSize = 100;
    int maxHeight = 3;
    int numGenerations = 50;
    double mutationProbability = 0.1;

    // Generate the initial population
    List<ExpNode> population = new ArrayList<>();
    for (int i = 0; i < populationSize; i++) {
        population.add(randomExpression(maxHeight));
    }

    // Run the genetic algorithm
    for (int generation = 1; generation <= numGenerations; generation++) {
        // Evaluate the fitness of each individual
        Map<ExpNode, Double> fitnesses = new HashMap<>();
        for (ExpNode individual : population) {
            double fitness = 0.0;
            for (int i = 0; i < input.length; i++) {
                double actual = actualOutput(input[i]);
                double predicted = individual.value(input[i]);
                fitness += Math.pow(actual - predicted, 2);
            }
            fitness /= input.length;
            fitnesses.put(individual, fitness);
        }

        // Select the parents for the next generation
        List<ExpNode> parents = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            ExpNode parent1 = tournamentSelection(population, fitnesses);
            ExpNode parent2 = tournamentSelection(population, fitnesses);
            parents.add(parent1);
            parents.add(parent2);
        }

        // Create the next generation
        List<ExpNode> nextGeneration = new ArrayList<>();
        for (int i = 0; i < populationSize; i += 2) {
            ExpNode parent1 = parents.get(i);
            ExpNode parent2 = parents.get(i + 1);
            ExpNode child1 = parent1.copy();
            ExpNode child2 = parent2.copy();
            if (Math.random() < mutationProbability) {
                subtreeMutation(child1, maxHeight);
            }
            if (Math.random() < mutationProbability) {
                subtreeMutation(child2, maxHeight);
            }
            onePointCrossover(child1, child2, maxHeight);
            nextGeneration.add(child1);
            nextGeneration.add(child2);
        }
        population = nextGeneration;
    }

   
  
    // Find the best individual in the final population
      ExpNode bestIndividual = null;
      double bestFitness = Double.POSITIVE_INFINITY;
      Map<ExpNode, Double> fitnesses = new HashMap<>();
      for (ExpNode individual : population) {
          double fitness = 0.0;
          for (int i = 0; i < input.length; i++) {
              double actual = actualOutput(input[i]);
              double predicted = individual.value(input[i]);
              fitness += Math.pow(actual - predicted, 2);
          }
          fitness /= input.length; // calculate the mean squared error
          fitnesses.put(individual, fitness);
          if (fitness < bestFitness) {
              bestFitness = fitness;
              bestIndividual = individual;
          }
      }
      System.out.println("Best individual: " + bestIndividual.toString());
      System.out.println("Fitness: " + bestFitness);


    }

    //node classes
    static abstract class ExpNode {
      ExpNode child;
    // Other fields and methods

    public int getHeight() {
        if (child == null) {
            return 0;
        } else {
            return 1 + child.getHeight();
        }
    }
        abstract double value(double x);
        abstract ExpNode copy();
        abstract void replaceWith(ExpNode node);

    }

    static class ConstNode extends ExpNode {
      ExpNode child;
      void replaceWith(ExpNode node) {
        child = node;
    }
    
    
        double value;
        ConstNode(double val) {
            value = val;
        }
        double value(double x) {
            return value;
        }
        ExpNode copy() {
            return new ConstNode(value);
        }
        public String toString() {
            return String.format("%.2f", value);
        }
    }

    static class VariableNode extends ExpNode {
      ExpNode child;
      void replaceWith(ExpNode node) {
        child = node;
    }
    
        double value(double x) {
            return x;
        }
        ExpNode copy() {
            return new VariableNode();
        }
        public String toString() {
            return "x";
        }
    }

    static class BinaryOpNode extends ExpNode {
      void replaceWith(ExpNode node) {
        if (Math.random() < 0.5) {
            left = node;
        } else {
            right = node;
        }
    }
    
        char operator;
        ExpNode left, right;
        BinaryOpNode(char op, ExpNode l, ExpNode r) {
            operator = op;
            left = l;
            right = r;
        }
        double value(double x) {
            double leftVal = left.value(x);
            double rightVal = right.value(x);
            switch (operator) {
                case '+':
                    return leftVal + rightVal;
                case '-':
                    return leftVal - rightVal;
                case '*':
                    return leftVal * rightVal;
                case '/':
                    return leftVal / rightVal;
                case '^':
                    return Math.pow(leftVal, rightVal);
                default:
                    return Double.NaN;
            }
        }
        ExpNode copy() {
            return new BinaryOpNode(operator, left.copy(), right.copy());
        }
        public String toString() {
            String leftStr = left instanceof BinaryOpNode ? "(" + left.toString() + ")" : left.toString();
            String rightStr = right instanceof BinaryOpNode ? "(" + right.toString() + ")" : right.toString();
            return leftStr + " " + operator + " " + rightStr;
        }
    }

    static class UnaryOpNode extends ExpNode {
      void replaceWith(ExpNode node) {
        child = node;
    }
    
    
        int function;
        ExpNode child;

        UnaryOpNode(int func, ExpNode c) {
            function = func;
            child = c;
        }

        double value(double x) {
            double childVal = child.value(x);
            switch (function) {
                case 0:
                    return Math.sin(childVal);
                case 1:
                    return Math.cos(childVal);
                case 2:
                    return Math.exp(childVal);
                case 3:
                    return Math.abs(childVal);
                case 4:
                    return -childVal;
                default:
                    return Double.NaN;
            }
        }

        ExpNode copy() {
            return new UnaryOpNode(function, child.copy());
        }

        public String toString() {
            String childStr = child instanceof BinaryOpNode ? "(" + child.toString() + ")" : child.toString();
            switch (function) {
                case 0:
                    return "sin(" + childStr + ")";
                case 1:
                    return "cos(" + childStr + ")";
                case 2:
                    return "exp(" + childStr + ")";
                case 3:
                    return "abs(" + childStr + ")";
                case 4:
                    return "-" + childStr;
                default:
                    return "NaN";
            }
        }
    }
  

    // generate a random expression tree of a given height
    static ExpNode randomExpression(int height) {
        if (height == 0) {
            // base case: return a random constant or variable
            if (Math.random() < 0.5) {
                return new ConstNode(Math.random() * 10 - 5);
            } else {
                return new VariableNode();
            }
        } else {
            // recursive case: return a random binary operator with random children
            char[] operators = {'+', '-', '*', '/', '^'};
            char op = operators[(int) (Math.random() * operators.length)];
            ExpNode left = randomExpression(height - 1);
            ExpNode right = randomExpression(height - 1);
            return new BinaryOpNode(op, left, right);
        }
    }
}