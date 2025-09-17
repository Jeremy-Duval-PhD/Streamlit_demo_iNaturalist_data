Exporté à 2025-09-05T08:08:42Z

Requête: quality_grade=any&identifications=any&place_id=6753&taxon_id=63134&verifiable=true&spam=false

Colonnes:
id: Identifiant unique et séquentiel pour une observation.
uuid: Identificateur unique universel pour l’observation. Voir https://datatracker.ietf.org/doc/html/rfc9562
observed_on_string: Horodatage entré par l’observateur
observed_on: Date normalisée de l’observation
time_observed_at: Horodatage normalisé de l’observation
time_zone: Fuseau horaire de l’observation
user_id: Identifiant unique et séquentiel pour l'observateur
user_login: Indicatif ou pseudonyme de l'observateur, par ex. un identifiant court, unique, alphanumérique pour un utilisateur
user_name: Nom de l'observateur, généralement utilisé pour l'attribution sur le site. C'est un champ optionnel et peut être un pseudonyme.
created_at: Horodatage de la création de l’observation
updated_at: Horodatage de la dernière mise à jour de l’observation
quality_grade: Niveau de qualité de cette observation. Consultez la section Aide pour plus de détails sur ce que cela signifie. Voir https://help.inaturalist.org/support/solutions/articles/151000169936
license: Identifiant pour la licence ou renonciation que l'observateur a choisi pour cette observation. Tous droits réservés si vide.
url: URL de l’observation
image_url: URL de la première photo associée avec l'observation
sound_url: URL du premier son associé avec l'observation. Notez que cela ne sera présent que pour les téléversements directs, mais pas les sons hébergés sur des services tiers.
tag_list: Liste d’étiquettes séparée par des virgules
description: Texte écrit par l'observateur décrivant l'observation ou l'enregistrement ou toute autre note semblant pertinente
num_identification_agreements: Nombre d'identifications associées à un taxon qui correspondent ou sont contenues dans le taxon à partir de l'identification de l'observateur
num_identification_disagreements: Nombre d'identifications associées à un taxon qui ne correspondent pas ou ne pas sont contenues dans le taxon à partir de l'identification de l'observateur
captive_cultivated: Il s'agit de l'observation d'un organisme à ce moment et à cet endroit, parce que l'homme l'a volontairement placé là. Pour plus d'information, voir https://help.inaturalist.org/support/solutions/articles/151000169932
oauth_application_id: Identifiant séquentiel de l'application qui a créé l'observation. Pour plus d'informations à propos de l'application, ajouter ce nombre après https://www.inaturalist.org/oauth/applications/
place_guess: Description de l’emplacement entré par l’utilisateur
latitude: Latitude visible publiquement à partir de la localisation de l'observation
longitude: Longitude visible publiquement à partir de la localisation de l'observation
positional_accuracy: Exactitude des coordonnées (oui, oui, exactitude ≠ précision, mauvais choix de noms)
private_place_guess: Emplacement de l’observation tel qu’il est écrit par l’observateur si l’emplacement est masqué
private_latitude: Latitude privée, établie si l’observation est privée ou masquée
private_longitude: Longitude privée, établie si l’observation est privée ou masquée
public_positional_accuracy: Incertitude positionnelle maximale en mètres; inclut l'incertitude ajoutée par le masquage des coordonnées
geoprivacy: Si l'observateur a choisi ou non de masquer les coordonnées. Voir https://help.inaturalist.org/support/solutions/articles/151000169938
taxon_geoprivacy: La géoconfidentialité la plus conservatrice est appliquée en raison du statut de conservation des taxons selon l’identification actuelle.
coordinates_obscured: Si les coordonnées ont été masquées, c'est soit à cause de la géoconfidentialité, soit à cause d’un taxon menacé
positioning_method: Façon dont les coordonnées ont été déterminées
positioning_device: Appareil utilisé pour déterminer les coordonnées
species_guess: Nom du taxon observé en texte brut ; peut être défini par l'observateur lors de la création de l'observation, mais peut être remplacé par des noms canoniques et localisés quand le taxon change.
scientific_name: Nom scientifique du taxon observé selon iNaturalist
common_name: Nom commun ou vernaculaire du taxon observé selon iNaturalist
iconic_taxon_name: Catégorie taxinomique de rang supérieur pour le taxon observé
taxon_id: Identifiant unique et séquentiel pour le taxon observé

Pour plus d'informations sur les en-têtes de colonnes, voir https://www.inaturalist.org/terminology

