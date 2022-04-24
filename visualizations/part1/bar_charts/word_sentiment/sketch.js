
// Set the dimensions and margins of the graph
const SCREEN_DIMENSIONS = { 
    width: 850,
    height: 400,
    leftMargin: 10,
    rightMargin: 115,
    topMargin: 10,
    bottomMargin: 100,
    cellPadding: 20
};
SCREEN_DIMENSIONS.innerWidth = SCREEN_DIMENSIONS.width - SCREEN_DIMENSIONS.leftMargin - SCREEN_DIMENSIONS.rightMargin;
SCREEN_DIMENSIONS.innerHeight = SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.topMargin - SCREEN_DIMENSIONS.bottomMargin;


// Run `cleanDataOutput` in `file_manager.py` to ensure most recent data
const DATA_LOC = '../../data/words_sentiment_distrib.csv';
const TITLE = "Sentiment Distribution of Words";

// Append the svg object to the body of the page
var svg = d3.select('body').append('svg')
            .attr('width', SCREEN_DIMENSIONS.width)
            .attr('height', SCREEN_DIMENSIONS.height)
            .attr("viewBox", [-SCREEN_DIMENSIONS.leftMargin, -SCREEN_DIMENSIONS.topMargin, 
                                SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height])
            .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

const xParams = {
    value: row => row.percent,
    scale: d3.scaleLinear()
};
xParams.axis = d3.axisLeft()
    .scale(xParams.scale)

const colorParams = {
    value: row => row.sentiment
} 


const canvas = svg.append('g')
    .attr('transform', `translate(${SCREEN_DIMENSIONS.leftMargin},${SCREEN_DIMENSIONS.topMargin})`);

const xAxisGroup = canvas.append('g')
    .attr('transform', `translate(0, ${SCREEN_DIMENSIONS.innerHeight})`);

const titleGroup = canvas.append("text")
    .attr('transform', `translate(${(SCREEN_DIMENSIONS.innerWidth / 2)},${0 - (SCREEN_DIMENSIONS.topMargin)})`)
    .attr("class", "title")
    .text(TITLE);

const legendGroup = canvas.append('g')
    .attr("transform", `translate(${SCREEN_DIMENSIONS.innerWidth + SCREEN_DIMENSIONS.rightMargin / 5},${SCREEN_DIMENSIONS.topMargin * 2})`)
    .attr("class", "legendOrdinal");


xAxisGroup.append('text')
    .attr('class', 'x-axis-label')
    .attr('x', SCREEN_DIMENSIONS.innerWidth / 2)
    .attr('y', 50)
    .text(xParams.label);


const preprocess = (row, i) => {
    // Know there are no null-values from preprocessing
    return {
        'sentiment': row['sentiment'],
        'count': parseInt(row['count']),
        'percent': parseFloat(row['percent'])
    }
};

const percentRanges = arr => {
    var scaled = []
    let carryover = 0
    for (let i = 0; i < arr.length; ++i){
        scaled.push({
            'sentiment': arr[i].sentiment, 
            'percent': arr[i].percent,
            'range': [carryover, arr[i].percent + carryover]
        })
        carryover += arr[i].percent
    }
    return scaled
}


d3.csv(DATA_LOC, preprocess).then((data, i) => {


    xParams.scale
        .domain([0, 100])
        .range([0, SCREEN_DIMENSIONS.innerWidth]);
    

    var colorScale = d3.scaleOrdinal()
                    .domain(['negative', 'positive', 'neutral'])
                    .range(['#e41a1c', '#4daf4a', '#377eb8'])
    

    var ranges = percentRanges(data);

    var bars = canvas.selectAll('rect')
        .data(ranges)
        .enter()

    const constantY = SCREEN_DIMENSIONS.topMargin + SCREEN_DIMENSIONS.innerHeight / 3,
            barHeight = 50;

    // Build rectangles
    bars.append('rect')
        .attr('class', 'bar')
        .attr('x', d => xParams.scale(d.range[0]))
        .attr('y', d => constantY)
        .attr('width', d => xParams.scale(d.range[1] - d.range[0]))
        .attr('height', barHeight)
        .attr('fill', d => colorScale(colorParams.value(d)))

    // Add percentages in bars
    bars.append("text")
        .attr('class', 'percent-label')
        .attr("x", d => xParams.scale(d.range[0]) + (xParams.scale(d.range[1] - d.range[0]) / 2) - 17)
        .attr("y", d => constantY + barHeight * 0.55)
        .text(d => d3.format(".1%")(d.percent / 100));  

    
    canvas.append('rect')
            .attr('class', 'legend-border')
            .attr('x', SCREEN_DIMENSIONS.leftMargin + SCREEN_DIMENSIONS.innerWidth)
            .attr('y', SCREEN_DIMENSIONS.topMargin - 10)
            .attr('width', d => 100)
            .attr('height', 100)
    


    // Define and build legend
    var legend = d3.legendColor()
        .shapeHeight(15)
        .orient("vertical")
        .scale(colorScale) 
        .title("Sentiment")
        .titleWidth(30);

    legendGroup.call(legend);
});