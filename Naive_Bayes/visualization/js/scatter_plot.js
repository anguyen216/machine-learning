function scatter_plot(data) {
	// create tooltip div
	var tooltip = d3.select('#scatter-plot').append('div')
					.attr('class', 'tooltip');
	// set up svg area
	var svg = d3.select('#scatter-plot')
				.append('svg')
				.attr('width', mainW)
				.attr('height', mainH);

	var width = mainW - margin.left - margin.right,
		height = mainH - margin.top - margin.bottom;

	svg.append('g')
		.attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

	// scaling functions
	var x_scale = d3.scaleLinear()
					.range([0, width])
					.domain(d3.extent(data, function(d) {
						return d.Polarity;
					})).nice();

	var y_scale = d3.scaleLinear()
					.range([height - 10, 0])  // because y-axis increase as we go down :)
					.domain([0, d3.max(data, function(d) {
						return d.Count;
					})]).nice();

	// dot size
	var radius = d3.scaleSqrt()
					.range([2, 10])
					.domain(d3.extent(data, function(d) {
						return Math.abs(d.Polarity);
					})).nice();

	// config axes
	var x_axis = d3.axisBottom()
					.scale(x_scale);
	var y_axis = d3.axisLeft()
					.scale(y_scale);
	var color = d3.scaleOrdinal(['#F57C00', 'grey', 'steelblue']);

	// graph
	// adding axes
	svg.append('g')
		.attr('transform', 'translate(50,' + height + ')')
		.attr('class', 'x-axis')
		.call(x_axis);
	svg.append('g')
		.attr('transform', 'translate(50,10)')
		.attr('class', 'y-axis')
		.call(y_axis);

	// adding data points
	var dot = svg.selectAll('.dot')
					.data(data)
					.enter().append('circle')
					.attr('class', 'dot')
					.attr('cx', function(d){
						return x_scale(d.Polarity) + 50;  // shift point along with x-axis
					})
					.attr('cy', function(d){
						return y_scale(d.Count) + 10;  // shift point along with y-axis
					})
					.attr('r', function(d){
						return radius(Math.abs(d.Polarity));
					})
					.style('fill', function(d){
						return color(d.Group);
					})
					.on('mousemove', function(d) {
						// ths isn't the best :(, but I'm trying :)
						tooltip.style('left', d3.event.pageX + 10 + 'px')
								.style('top', d3.event.pageY - 20 + 'px')
								.style('display', 'block')
								.style('opacity', 0.8)
								.html(d.Word);
					})
					.on('mouseout', function(d) {
						tooltip.style('display', 'none');
					});

	// add axes labels
	svg.append('text')
		.attr('transform', 'translate(50, 10)')
		.attr('x', 10)
		.attr('y', 10)
		.attr('class', 'label')
		.text('Frequency');

	svg.append('text')
		.attr('x', width + 50)
		.attr('y', height - 10)
		.attr('text-anchor', 'end')
		.attr('class', 'label')
		.text('Polarity');

}