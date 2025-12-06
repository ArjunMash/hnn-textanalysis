# Data Dictionary for [all_articles_first_week](data/all_articles_first_week.csv)
| Field | Description |
| --- | --- |
| url | The canonical URL for a page on your site, with no appended parameters. |
| publishDate | The calendar date on which the article was published. This determines the recency score of your content in Discover. For example, an article with higher recency score might be prioritized over one that has higher relevance score. Visit the Discover documentation to learn more about article publish date. |
| uniqueUsers | The total number of unique users that have visited your site |
| pageViewsTotal | The total number of pages viewed by users visiting your site. |
| totalEngagementTime | Total time spent by users actively engaged on your site. Marked by a 10-second period where the users focus is on your site, at least one event is triggered, or the first page is loaded. Visit the Engagement Time documentation for a more detailed breakdown of Engagement Time. |
| totalDuration | The total amount of time users spent reading your article after clicking on it from a Discover card. This is a legacy metric, you should use Total engagement time instead. |
| discoverImpressions | The total number of times a discover card is shown to users. |
| googleNewsImpressions | The total number of times a link to your site appears in Google News. |
| searchImpressions | The total number of links to your site a user has seen in their search results. |
| searchClicks | The total number of clicks from a Google search results page that have landed on your site. Learn more About impressions, position, and clicks. |
| discoverClicks | The total number of clicks on your Google Discover card. |
| googleNewsClicks | The total number of clicks from Google News that have landed on your site. |
| averageSearchResultPosition | The average position of your website’s links in the Google search results page. For example, if your link is always in the first position, your average position is 1. |
| earningsEstimate | Estimated earnings for the specified reporting date. This amount is an estimate that can change before the end of the month. |
| sessions | A session is a group of user interactions with your website that take place within a given time frame. For example, a single session can contain multiple page views. |
| sessionDuration | The average duration of users sessions on your website. |
| returningUsers | The number of users who visited your website in the past and visited again between the selected reporting dates. |
| newUsers | The number of users who visited your website for the first time between the selected reporting dates. |
| returningAudience | The returning audience for each page of your site and each source between the selected reporting dates. |
| newAudience | The new audience for each page of your site and each source between the selected reporting dates. |
| videoStarts | The total number of times users started playing your videos. |
| newUsersPerc | The percentage of new users across all users visiting your site between the selected reporting dates. |
| discoverCtr | The click-through rate of your discover cards, calculated as the number of clicks divided by the number of impressions for your discover cards. |
| googleNewsCtr | The click-through rate of your site in Google News, calculated as the number of clicks divided by the number of impressions for your site. |
| searchCtr | The click-through rate of your site on the Google search results page, calculated as the number of clicks divided by the number of impressions for your site. |
| discoverPosition | The average position of your content on Discover cards. |
| googleNewsPosition | The average position of your content on Google News. |
| searchPosition | The average position of your content on the Google search results page. |
| newsletterSubscribers | The total count of readers who subscribe to your newsletter from a particular page on your site, as captured by the newsletter_subscribe event, between the selected reporting dates. |
| covered | A boolean value, indicating whether we have complete CLS and LCP metrics for a URL. |
| fieldCls | A floating point value for the Cumulative Layout Shift metric, as captured from your user field data. |
| fieldLcp | A floating point value for the Largest Contentful Paint metric, as captured from your user field data. |
| sacOrganicVisits | The number of organic search visits to your site. |
| sacReferralVisits | The number of referral visits to your site. |
| sacVisits | The total number of organic search and referral visits users make to your site. This sum excludes any organic search or referral visits resulting from site visitors clicking on your links in Google’s search results. |