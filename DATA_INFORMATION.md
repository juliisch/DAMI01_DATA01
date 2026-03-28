# Variablenbeschreibung

## Transaktionsdaten

| Variablenname     | Beschreibung                                    |
|------------------|--------------------------------------------------|
| id               | Transaktions-Identifikationsnummer                |
| date             | Datum                                            |
| client_id        | Kundennummer                                     |
| card_id          | Kartennummer                                     |
| amount           | Transaktionsbetrag                               |
| use_chip         | Kennzeichen, ob Chip verwendet worden ist        |
| merchant_id      | Händler-Identifikationsnummer                    |
| merchant_city    | Stadt des Händlers                               |
| merchant_state   | Staat des Händlers                               |
| zip              | Postleitzahl des Händlers                        |
| mcc              | Merchant Category Code des Händlers              |
| errors           | Fehler bei der Transaktionsverarbeitung          |


## Kartendaten

| Variablenname           | Beschreibung                                       |
|-------------------------|----------------------------------------------------|
| id                      | Identifikationsnummer                              |
| client_id               | Kundennummer                                       |
| card_brand              | Kartenmarke                                        |
| card_type               | Kartentyp                                          |
| card_number             | Kartennummer                                       |
| expires                 | Ablaufdatum der Karte                              |
| cvv                     | Sicherheitscode der Karte                          |
| has_chip                | Kennzeichnung, ob Karte einen Chip besitzt         |
| num_cards_issued        | Anzahl der Karten pro Konto                        |
| credit_limit            | Kreditrahmen der Karte                             |
| acct_open_date          | Datum der Karteneröffnung                          |
| year_pin_last_changed   | Jahr der letzten Pin-Änderung                      |
| card_on_dark_web        | Kennzeichen, ob Karte im Darkweb gefunden wurde    |


## Kundendaten

| Variablenname        | Beschreibung                                     |
|----------------------|--------------------------------------------------|
| id                   | Kunden-Identifikationsnummer                     |
| current_age          | Aktuelles Alter des Kunden                       |
| retirement_age       | Geplantes Renteneintrittsalter des Kunden        |
| birth_year           | Geburtsjahr des Kunden                           |
| birth_month          | Geburtsmonat des Kunden                          |
| gender               | Geschlecht des Kunden                            |
| address              | Adresse des Kunden                               |
| latitude             | Breitengrad des Wohnorts                         |
| longitude            | Längengrad des Wohnorts                          |
| per_capita_income    | Pro-Kopf-Einkommen im Wohngebiet                 |
| yearly_income        | Jährliches Bruttoeinkommen des Kunden            |
| total_debt           | Gesamtschulden des Kunden                        |
| credit_score         | Bonitätsscore                                    |
| num_credit_cards     | Anzahl der Kreditkarten pro Kunde                |


## MCC-Daten

| Variablenname | Beschreibung                                         |
|---------------|------------------------------------------------------|
| mcc           | Merchant Category Code (Branchenklassifikationscode) |
| description   | Beschreibung des Merchant Category Code              |


**Quelle**: Quelle: Vazquez, V. V. Financial Transactions Dataset: Analytics. Abgerufen am 20.02.2026 von https://www.kaggle.com/datasets/computingvictor/transactions-fraud-datasets
