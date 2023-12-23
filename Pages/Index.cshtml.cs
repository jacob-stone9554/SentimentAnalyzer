using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.RazorPages;
using SentimentAnalyzerWebApp.ML;

namespace SentimentAnalyzerWebApp.Pages
{
    public class IndexModel : PageModel
    {
        [BindProperty]
        public string? input { get; set; }

        public SentimentData sentimentData { get; set; }
        public SentimentPrediction prediction { get; set; }
        public MLHandler mlHandler { get; set; }
        public string result;
        public string test { get; set; }

        private readonly ILogger<IndexModel> _logger;

        public IndexModel(ILogger<IndexModel> logger)
        {
            _logger = logger;
            sentimentData = new SentimentData();
            mlHandler = new MLHandler();
            prediction = new SentimentPrediction();
        }

        public void OnGet()
        {

        }

        public IActionResult OnPost()
        {
            sentimentData.SentimentText = input;

            

            result = mlHandler.UseModelWithSingleItem(sentimentData);


            return Page();
        }
    }
}


/*
 * 
 * you need to:
 * ->bind an input string to string variable.
 * ->create an instance of SentimentData.
 * ->assign that string to SentimentData.SentimentText
 * ->create an instance of ML Handler
 * ->call all of the necessary functions to train the ML model (did so in the constructor)
 * ->call UseModelWithSingleItem, passing in the SentimentData previously created
 * ->assign the result of UseModelWithSingleItem to a SentimentPrediction
 * ->Print that sentiment prediction on the cshtml view
 */