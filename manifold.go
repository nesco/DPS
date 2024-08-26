package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
)

// Bottom level data: Task Object (Each input/output examples of 2D grid of colors)

type Task struct {
	Input [][]int `json:"input"`
	Output [][]int `json:"output"`
}

// Top level of the JSON: The train tasks /examples and the test ones
type Data struct {
	Train []Task `json:"train"`
	Test []Task `json:"test"`
}

type Square [9]int

type SquareCount struct {
	Square Square
	Count int
}

func extractSquares(matrix [][]int, squareMap map[Square]int) {
	// Building a map/dict keys: square vectors value: colors

	numRows := len(matrix)
	numCols := len(matrix[0])

	if numRows < 3 || numCols < 3 {
		return
	}

	// Extraction
	for i := 1; i <= numRows - 2; i++ {
		for j := 1; j <= numCols - 2; j++ {
			var locSquare Square
			index := 0 // current local square index
			for k := 0; k < 3; k++ {
				for l := 0; l < 3; l++ {
					locSquare[index] = matrix[i-1+k][j-1+l] // i-1, i, j+1 X j-1, j, j+1
					index++
				}
			}
			// Adding +1 to the local square count
			squareMap[locSquare]++
		}
	}
}

func extractCounts(squareMap map[Square]int) []SquareCount {
	// Converting the Map into a slice of tuples
	var result []SquareCount
	for square, count := range squareMap {
		result = append(result, SquareCount{Square: square, Count: count})
	}
	return result
}

func main() {

	// reading a json
	data, err := ioutil.ReadFile("../data/training/0a938d79.json")

	if err != nil {
		log.Fatalf("Error reading file: %s", err)
	}

	var dataSet Data
	err = json.Unmarshal(data, &dataSet)

	if err != nil {
		log.Fatalf("Error decoding JSON: %s", err)
	}

	/*if len(dataSet.Train) > 0 {
		fmt.Println("First input example: ", dataSet.Train[0].Input)
		fmt.Println("First output example: ", dataSet.Train[0].Output)
		}*/

	// Creating document frequency
	examplesLen := len(dataSet.Train)
	squares := make(map[Square]int)

	for i := 0; i <examplesLen; i++ {
		extractSquares(dataSet.Train[i].Output, squares)
	}

	counts := extractCounts(squares)

    // Print all extracted 3x3 squares and their counts
    // LocalMaps in an Atlas?
    for _, sc := range counts {
        fmt.Printf("Square: %+v, Count: %d\n", sc.Square, sc.Count)
    }
}
