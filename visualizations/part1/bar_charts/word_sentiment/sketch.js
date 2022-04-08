


// set the dimensions and margins of the graph
const SCREEN_DIMENSIONS = { 
    width: 600,
    height: 600,
    leftMargin: 30,
    rightMargin: 115,
    topMargin: 10,
    bottomMargin: 100,
    cellPadding: 20
};
SCREEN_DIMENSIONS.innerWidth = SCREEN_DIMENSIONS.width - SCREEN_DIMENSIONS.leftMargin - SCREEN_DIMENSIONS.rightMargin;
SCREEN_DIMENSIONS.innerHeight = SCREEN_DIMENSIONS.height - SCREEN_DIMENSIONS.topMargin - SCREEN_DIMENSIONS.bottomMargin;


// Run `cleanDataOutput` in `file_manager.py` to ensure most recent data
// const DATA_LOC = '../data/slava_ukraine_sentiment_transform.csv';
const DATA_LOC = '../../data/words_summary.csv';

// Append the svg object to the body of the page
var svg = d3.select('body').append('svg')
            .attr('width', SCREEN_DIMENSIONS.width)
            .attr('height', SCREEN_DIMENSIONS.height)
            .attr("viewBox", [-SCREEN_DIMENSIONS.leftMargin, -SCREEN_DIMENSIONS.topMargin, 
                                SCREEN_DIMENSIONS.width, SCREEN_DIMENSIONS.height])
            .attr("style", "max-width: 100%; height: auto; height: intrinsic;");

const targetCols = ['sentiments']


const xParams = {
    value: row => row.percent,
    label: 'Sentiment',
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
            'range': [carryover, arr[i].percent + carryover]
        })
        carryover += arr[i].percent
    }
    return scaled
}


d3.csv(DATA_LOC, preprocess).then((data, i) => {

    console.log(data);

    xParams.scale
        .domain([0, 100])
        .range([SCREEN_DIMENSIONS.innerHeight, 0]);
    

    var colorScale = d3.scaleOrdinal()
                    .domain(['negative', 'neutral', 'positive'])
                    .range(['#e41a1c','#377eb8','#4daf4a'])
    

    var ranges = percentRanges(data);

    console.log(ranges)

    canvas.selectAll('rect')
        .data(ranges)
        .enter()
        .append('rect')
        .attr('class', 'bar')
        .attr('x', d => xParams.scale(d.range[0]))
        .attr('y', d => SCREEN_DIMENSIONS.topMargin + SCREEN_DIMENSIONS.innerHeight / 4)
        .attr('width', d => xParams.scale(d.range[1] - d.range[0]))
        .attr('height', 50)
        .attr('fill', d => colorScale(colorParams.value(d)))
        .append("text")
                .attr("x", d => xParams.scale(d.range[1] - d.range[0]))
                .attr("y", d => SCREEN_DIMENSIONS.width / 2)
                .style("text-anchor", "middle")
                .style("font-size", "10px")
                .style("color", "white")
                .text('HERE');  

});