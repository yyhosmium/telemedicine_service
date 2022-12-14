let placesSearchCB = function(result, status) {
    if (status === kakao.maps.services.Status.OK) {
        document.getElementById('hospital').innerHTML=result[0].place_name;
    }
};
let geocoder = new kakao.maps.services.Geocoder();
geocoder.addressSearch('서울특별시 광운로 20', function(result, status) {
    if (status === kakao.maps.services.Status.OK) {
        let ps = new kakao.maps.services.Places();
        const keyword = '내과';
        ps.keywordSearch(keyword,placesSearchCB,{
            location: new kakao.maps.LatLng(result[0]['address'].y, result[0]['address'].x),
            sort: kakao.maps.services.SortBy.DISTANCE
        });
    }
});