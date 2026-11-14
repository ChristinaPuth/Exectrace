# Manual citation review (M5, manual test MT-5.6)

Ten answers from the last `leadership eval --chat`, five per company, including counts, locations and a
former executive (SPEC section 7.2). For each answer, open every cited URL and answer the three questions:

1. **Supports:** does the source say what the answer claims?
2. **Right person:** is it about the right person and company?
3. **Date and status:** does the answer convey the date and status correctly?

Write Yes or No, and a note when No. If a live page changed since it was retrieved (2026-11-13), write
"page changed" in the note instead of No. The result is reported as "N of 10" answers with three Yes.

## 1. robinhood.com — Who is the CEO? (sample question)

Answer, as the user sees it:

> Vladimir Tenev is the Chair and Chief Executive Officer of Robinhood, according to a SEC filing dated 2026‑04‑22: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm.

Cited URLs: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | SEC filing confirms Vladimir Tenev as Chair and CEO. |

## 2. robinhood.com — Who are the current C-level executives? (sample question, several people)

Answer, as the user sees it:

> The current C‑level team at Robinhood (as of the latest public filings) includes:
>
> - **Vladimir Tenev** – Chair and Chief Executive Officer (SEC filing, 2026‑04‑22) ; also listed as President (SEC filing, 2026‑02‑18).  
> - **Daniel Gallagher** – Chief Legal Officer (SEC filing, 2026‑04‑22) ; also titled Chief Legal, Compliance, and Corporate Affairs Officer (same source).  
> - **Dara Bazzano** – Chief Accounting Officer (SEC filing, 2026‑06‑26).  
> - **Shiv Verma** – Chief Financial Officer (SEC filing, 2026‑06‑26).  
> - **Steven Quirk** – Chief Brokerage Officer (SEC filing, 2026‑04‑22).  
>
> These five individuals are the C‑level executives recorded in the current leadership data.
>
> Sources:
> - Daniel Gallagher: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm (sec_filing, 2026-04-22)
> - Dara Bazzano: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000083/hood-20260625.htm (sec_filing, 2026-06-26)
> - Shiv Verma: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000083/hood-20260625.htm (sec_filing, 2026-06-26)
> - Steven Quirk: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm (sec_filing, 2026-04-22)
> - Vladimir Tenev: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm (sec_filing, 2026-04-22)

Cited URLs: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm, https://www.sec.gov/Archives/edgar/data/1783879/000178387926000083/hood-20260625.htm

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | The cited SEC filings support all five executives and their titles. The list is limited to collected data. |

## 3. robinhood.com — How many C-level executives were found? (a count)

Answer, as the user sees it:

> 5 found in public sources.
>
> Note: this counts only the people named in the sources that could be read. 29 sources found for this company could not be read (robots.txt or the site refused automated access), 15 of them on the company's own site. The real number may be higher.

Cited URLs: none (an answer that something was not found cites nothing)

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | Database confirms 5 distinct current C-level executives. The actual number may be higher due to unreadable sources. |

## 4. robinhood.com — Who was the previous CFO? (a former executive)

Answer, as the user sees it:

> The previous CFO was Jason Warnick, who ceased to serve as Chief Financial Officer effective 2026-02-06 and has left the role; according to a SEC filing, 2026-04-22: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm

Cited URLs: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm

(Changed after the first review: a 2020 press release about another executive's hire was appended as
Warnick's source although the answer already cited the SEC filing that ended his role. Fixed in
`leadership/citations.py`; regression test in `tests/test_chat.py`.)

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | SEC filing confirms Jason Warnick's CFO role ended on 2026-02-06. The unrelated citation was removed. |

## 5. robinhood.com — Where is the CEO based? (a location)

Answer, as the user sees it:

> The CEO is Vladimir Tenev, but a base location was not found in the collected data according to a sec_filing, 2026-04-22: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm

Cited URLs: https://www.sec.gov/Archives/edgar/data/1783879/000178387926000053/hood-20260422.htm

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | SEC confirms the CEO's identity; no location was found in the collected database. |

## 6. meetcampfire.com — Who is the CEO? (sample question)

Answer, as the user sees it:

> John Glasgow is listed as the CEO of Campfire according to a company site source dated 2026‑11‑13 (https://campfire.ai/basecamp).

Cited URLs: https://campfire.ai/basecamp

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | Official company page identifies John Glasgow as CEO and Founder. |

## 7. meetcampfire.com — Who leads finance? (a head of department)

Answer, as the user sees it:

> Erik Leavell leads finance as Head of Finance & Applied AI, according to company_site, 2026-11-13: https://campfire.ai/basecamp

Cited URLs: https://campfire.ai/basecamp

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | Official company page confirms Erik Leavell as Head of Finance & Applied AI. |

## 8. meetcampfire.com — Who is the VP of Sales? (a vice president)

Answer, as the user sees it:

> Chris McDonough is the VP of Sales, according to news, 2026-09-10: https://thegtmnewsletter.substack.com/p/gtm-205-ai-native-sales-organization-chris-mcdonough-campfire

Cited URLs: https://thegtmnewsletter.substack.com/p/gtm-205-ai-native-sales-organization-chris-mcdonough-campfire

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | Third-party interview identifies Chris McDonough as Campfire's VP of Sales. |

## 9. meetcampfire.com — How many heads of department were found? (a count)

Answer, as the user sees it:

> 1 found in public sources.
>
> Note: this counts only the people named in the sources that could be read. 3 sources found for this company could not be read (robots.txt or the site refused automated access). The real number may be higher.

Cited URLs: none (an answer that something was not found cites nothing)

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | Database confirms 1 current department head. The count may be incomplete. |

## 10. meetcampfire.com — Where is John Glasgow based? (a location)

Answer, as the user sees it:

> John Glasgow’s location was not found in the collected data. according to company_site, 2026-11-13: https://campfire.ai/basecamp

Cited URLs: https://campfire.ai/basecamp

| Supports | Right person | Date and status | Note |
|---|---|---|---|
| Yes | Yes | Yes | Official company page confirms his identity; no location was found in the collected database. |

## Result

| | |
|---|---|
| Answers with three Yes | 10 of 10 |
| Reviewed by / date | Project owner, 2026-11-13 |

Counts (items 3 and 9) and missing locations (items 5 and 10) were verified against the collected
database, not as complete real-world facts: a count covers only the sources that could be read, and
"no location found" means none was found in the collected sources.
