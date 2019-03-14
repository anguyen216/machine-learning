//===== SETUP =====
var mainW = 700;
var mainH = 400;
var margin = {top:20, right: 20, bottom: 30, left: 50};

//===== HELPERS =====
var rowConverter = function(d) {
	// convert csv data (usually d3 loads them as string) into correct data type
	return {
		Word: d.word,
		Count: +d.count,
		Polarity: +d.polarity,
		Group: d.group
	};
};

//===== MAIN =====
d3.csv('word_polarity_gh.csv', rowConverter, function(dataset) {
	console.log(dataset);  // debugging purpose
	scatter_plot(dataset);
})